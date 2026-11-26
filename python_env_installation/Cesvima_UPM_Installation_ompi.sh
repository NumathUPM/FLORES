#!/usr/bin/env bash
set -e

### ------------------------------------------------------------
### Environment modules (toolchain foss/2025a, GCC 14.2.0)
### ------------------------------------------------------------

MODULES="OpenMPI/5.0.7-GCC-14.2.0 OpenBLAS/0.3.29-GCC-14.2.0 ScaLAPACK/2.2.2-gompi-2025a-fb METIS/5.1.0-GCCcore-14.2.0 SCOTCH/7.0.8-gompi-2025a Python/3.13.1-GCCcore-14.2.0"

module purge
module load apps/avx512-2025
module load $MODULES

echo "MPI      : $EBROOTOPENMPI"
echo "OpenBLAS : $EBROOTOPENBLAS"
echo "ScaLAPACK: $EBROOTSCALAPACK"
echo "METIS    : $EBROOTMETIS"
echo "SCOTCH   : $EBROOTSCOTCH"
echo "GCC      : $(gcc --version | head -1)"
echo "Python   : $(which python3) ($(python3 --version 2>&1))"

### ------------------------------------------------------------
### 0. Project setup
### ------------------------------------------------------------

PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/myvenv_ompi"
PETSC_SRC="$PROJECT_DIR/petsc_ompi"
PETSC_ARCH="arch-linux-c-opt-ompi"
SLEPC_SRC="$PROJECT_DIR/slepc_ompi"
MUMPS_SRC="$PROJECT_DIR/mumps"
MUMPS_INSTALL="$PROJECT_DIR/mumps_install"
ENV_FILE="$PROJECT_DIR/load_env_flores.sh"

NPROC=$(nproc)

echo ">>> Project dir : $PROJECT_DIR"
echo ">>> NPROC       : $NPROC"

# --- Rutas de dependencias del sistema ---
MPI_DIR="$EBROOTOPENMPI"
OPENBLAS_DIR="$EBROOTOPENBLAS"
SCALAPACK_DIR="$EBROOTSCALAPACK"
METIS_DIR="$EBROOTMETIS"
SCOTCH_DIR="$EBROOTSCOTCH"

# --- Verificar que las variables del entorno estan definidas ---
for var in MPI_DIR OPENBLAS_DIR SCALAPACK_DIR METIS_DIR SCOTCH_DIR; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var no esta definida. Revisa los module load del principio"
        exit 1
    fi
    echo "    $var = ${!var}"
done

# --- Verificar que el Python cargado trae Python.h ---
PY_INCLUDE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [ ! -f "$PY_INCLUDE/Python.h" ]; then
    echo "ERROR: no se encuentra $PY_INCLUDE/Python.h"
    exit 1
fi
echo "    Python.h = $PY_INCLUDE/Python.h"


### ------------------------------------------------------------
### Checkpoint system
### ------------------------------------------------------------

CHECKPOINT_FILE="$PROJECT_DIR/.install_checkpoint"

checkpoint_done() {
    echo "$1" >> "$CHECKPOINT_FILE"
}

checkpoint_check() {
    [ -f "$CHECKPOINT_FILE" ] && grep -q "^$1$" "$CHECKPOINT_FILE"
}

# Mostrar checkpoints existentes
if [ -f "$CHECKPOINT_FILE" ]; then
    echo ">>> Checkpoints encontrados:"
    cat "$CHECKPOINT_FILE" | sed 's/^/    /'
fi

### ------------------------------------------------------------
### 0.1 Virtual environment
### ------------------------------------------------------------

# Un venv creado con otro Python (p. ej. el del sistema, sin Python.h) no sirve
if [ -f "$VENV_DIR/bin/activate" ]; then
    VENV_PY_INCLUDE="$("$VENV_DIR/bin/python3" -c 'import sysconfig; print(sysconfig.get_paths()["include"])' 2>/dev/null || true)"
    if [ ! -f "$VENV_PY_INCLUDE/Python.h" ]; then
        echo "ERROR: $VENV_DIR se creo con otro Python (sin cabeceras o roto)."
        echo "       Borralo (rm -rf $VENV_DIR) y vuelve a ejecutar el script."
        exit 1
    fi
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo ">>> Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo ">>> [SKIP] Virtual environment already exists"
fi

source "$VENV_DIR/bin/activate"
checkpoint_done "venv"

if ! python3 -c "import numpy, Cython" &>/dev/null; then
    echo ">>> Installing base Python packages"
    pip install --upgrade pip setuptools wheel numpy cython
else
    echo ">>> [SKIP] Base Python packages already installed"
fi

### ------------------------------------------------------------
### 1. MUMPS 5.5.1 con OpenMP + MPI + METIS + PT-Scotch
### ------------------------------------------------------------

if checkpoint_check "mumps" || [ -f "$MUMPS_INSTALL/lib/libzmumps.a" ]; then
    echo ">>> [SKIP] MUMPS already built (checkpoint)"
else
    echo ">>> Building MUMPS 5.5.1 with OpenMP + MPI + METIS + PT-Scotch"

    # Limpiar version anterior si existe
    if [ -d "$MUMPS_SRC" ]; then
        echo "    Removing previous MUMPS source..."
        rm -rf "$MUMPS_SRC"
    fi

    echo "    Downloading MUMPS 5.5.1..."
    wget -q https://mumps-solver.org/MUMPS_5.5.1.tar.gz -O /tmp/MUMPS_5.5.1.tar.gz
    tar xzf /tmp/MUMPS_5.5.1.tar.gz -C "$PROJECT_DIR"
    mv "$PROJECT_DIR/MUMPS_5.5.1" "$MUMPS_SRC"

    mkdir -p "$MUMPS_INSTALL"/{lib,include}

    cat > "$MUMPS_SRC/Makefile.inc" << EOF
# ── Compilers ────────────────────────────────────────────────
CC      = mpicc
FC      = mpif90
FL      = mpif90

# ── Output flag (necesario para gfortran >= 10) ──────────────
# Sin esto, gfortran interpreta el .o como fichero de entrada
OUTF    = -o
OUTC    = -o

# ── Optimization + OpenMP ────────────────────────────────────
# -fpermissive: GCC >= 14 trata como error codigo C antiguo
# (declaraciones implicitas, punteros incompatibles)
OPTF    = -O3 -fopenmp -march=native -funroll-loops -fallow-argument-mismatch -fPIC
OPTC    = -O3 -fopenmp -march=native -funroll-loops -fPIC -fpermissive
OPTL    = -fopenmp

# ── BLAS/LAPACK (OpenBLAS) ───────────────────────────────────
LIBBLAS = -L${OPENBLAS_DIR}/lib -lopenblas

# ── ScaLAPACK ────────────────────────────────────────────────
SCALAP  = -L${SCALAPACK_DIR}/lib -lscalapack

# ── METIS ─────────────────────────────────────────
IMETIS  = -I${METIS_DIR}/include
LMETIS  = -L${METIS_DIR}/lib -lmetis

# ── SCOTCH + PT-Scotch ───────────────────────────────────────
ISCOTCH = -I${SCOTCH_DIR}/include
LSCOTCH = -L${SCOTCH_DIR}/lib -lptesmumps -lptscotch -lptscotcherr -lscotch -lscotcherr

# ── MPI ──────────────────────────────────────────────────────
INCPAR  = -I${MPI_DIR}/include
LIBPAR  = \$(SCALAP) -L${MPI_DIR}/lib -lmpi_mpifh -lmpi

# ── Secuencial (fallback interno de MUMPS) ───────────────────
INCSEQ  = -I./libseq
LIBSEQ  = \$(LIBBLAS) -L./libseq -lmpiseq

INCS    = \$(INCPAR)
LIBS    = \$(LIBPAR)
LIBSEQNEEDED =

# ── Definiciones de preprocesador ────────────────────────────
CDEFS        = -DAdd_
OPTF        += -DBLR_MT

# ── Orderings ────────────────────────────────────────────────
ORDERINGSF   = -Dmetis -Dscotch -Dptscotch
ORDERINGSC   = \$(ORDERINGSF)
LORDERINGS   = \$(LMETIS) \$(LSCOTCH)
IORDERINGSF  = \$(IMETIS) \$(ISCOTCH)
IORDERINGSC  = \$(IMETIS) \$(ISCOTCH)

# ── Otras librerias ──────────────────────────────────────────
LIBOTHERS = -lpthread -lm

# ── Usar solo librerias estaticas (.a) — evita problemas con -lmumps_common ──
LIBEXT        = .a
LIBEXT_SHARED =
FPIC          = -fPIC
RANLIB        = ranlib
AR            = ar cr 
EOF


    # Step 1: PORD — los .c estan directamente en PORD/lib/
    echo "    Step 1/3: Compiling PORD (manual)..."
    cd "$MUMPS_SRC/PORD/lib"
    for src in *.c; do
        obj="${src%.c}.o"
        gcc -I../include -O3 -fopenmp -march=native -fPIC -fpermissive -c "$src" -o "$obj"
    done
    ar cr libpord.a *.o
    ranlib libpord.a
    cp libpord.a "$MUMPS_SRC/lib/"
    cd "$MUMPS_SRC"

    # Step 2: modulos comunes en serie (dependencias Fortran .mod)
    echo "    Step 2/3: Compiling common modules (serial)..."
    make -j1 -C src libcommon

    # Step 3: variantes s/d/c/z en paralelo
    echo "    Step 3/3: Compiling s/d/c/z variants (parallel j=$NPROC)..."
    make -j"$NPROC" -C src s d c z

    echo "    Installing MUMPS to $MUMPS_INSTALL..."
    cp lib/lib*.a "$MUMPS_INSTALL/lib/"
    cp include/*.h "$MUMPS_INSTALL/include/"

    echo "    MUMPS built OK"
    echo "    Libraries: $(ls $MUMPS_INSTALL/lib/)"
    checkpoint_done "mumps"
    cd "$PROJECT_DIR"
fi

### ------------------------------------------------------------
### 2. PETSc
### ------------------------------------------------------------

if [ ! -d "$PETSC_SRC/.git" ]; then
    echo ">>> Cloning PETSc"
    git clone -b release https://gitlab.com/petsc/petsc.git "$PETSC_SRC"
else
    echo ">>> [SKIP] PETSc source already cloned"
fi

export PETSC_DIR="$PETSC_SRC"
export PETSC_ARCH="$PETSC_ARCH"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"
export MPICC="$(which mpicc)"
mkdir -p "$PROJECT_DIR/.noreadline"; echo "raise ImportError" > "$PROJECT_DIR/.noreadline/readline.py"
export PYTHONPATH="$PROJECT_DIR/.noreadline${PYTHONPATH:+:$PYTHONPATH}"

if ! checkpoint_check "petsc_conf" && [ ! -f "$PETSC_SRC/$PETSC_ARCH/lib/petsc/conf/petscvariables" ]; then
    echo ">>> Configuring PETSc"
    cd "$PETSC_SRC"
    # Construir la lista de libs de MUMPS en una sola linea sin espacios ni saltos
    MUMPS_LIBS="[${MUMPS_INSTALL}/lib/libzmumps.a,${MUMPS_INSTALL}/lib/libmumps_common.a,${MUMPS_INSTALL}/lib/libpord.a,-L${SCALAPACK_DIR}/lib,-lscalapack,-L${OPENBLAS_DIR}/lib,-lopenblas,-L${METIS_DIR}/lib,-lmetis,-L${SCOTCH_DIR}/lib,-lptesmumps,-lptscotch,-lptscotcherr,-lscotch,-lscotcherr,-L${MPI_DIR}/lib,-lmpi_mpifh,-lmpi,-lgomp,-lpthread,-lm]"

    # METIS del modulo del sistema (la misma que usa MUMPS); los nodos no
    # tienen acceso a bitbucket.org, asi que no se usa --download-metis
    /usr/bin/python3 -X faulthandler ./configure \
      --PETSC_ARCH="$PETSC_ARCH" \
      --with-debugging=0 \
      --with-shared-libraries=1 \
      --with-mpi=1 \
      --with-scalar-type=complex \
      --with-mpi-dir="$MPI_DIR" \
      --with-blas-lapack-dir="$OPENBLAS_DIR" \
      --with-scalapack-dir="$SCALAPACK_DIR" \
      --with-mumps-include="${MUMPS_INSTALL}/include" \
      --with-mumps-lib="${MUMPS_LIBS}" \
      --with-metis-dir="$METIS_DIR" \
      COPTFLAGS="-O3 -march=native -fopenmp" \
      FOPTFLAGS="-O3 -march=native -fopenmp" \
      CXXOPTFLAGS="-O3 -march=native -fopenmp" \
      LDFLAGS="-fopenmp"
else
    echo ">>> [SKIP] PETSc already configured (checkpoint)"
fi
checkpoint_done "petsc_conf" 2>/dev/null || true

if ls "$PETSC_SRC/$PETSC_ARCH/lib/libpetsc.so."* 1>/dev/null 2>&1; then
    echo ">>> [SKIP] PETSc already built"
else
    echo ">>> Building PETSc"
    cd "$PETSC_SRC"
    make -j"$NPROC" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
    checkpoint_done "petsc"
fi

### ------------------------------------------------------------
### 3. SLEPc
### ------------------------------------------------------------

if [ ! -d "$SLEPC_SRC/.git" ]; then
    echo ">>> Cloning SLEPc"
    git clone -b release https://gitlab.com/slepc/slepc.git "$SLEPC_SRC"
else
    echo ">>> [SKIP] SLEPc source already cloned"
fi

export SLEPC_DIR="$SLEPC_SRC"
export LD_LIBRARY_PATH="$SLEPC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"

if [ ! -f "$SLEPC_SRC/$PETSC_ARCH/lib/slepc/conf/slepcvariables" ]; then
    echo ">>> Configuring SLEPc"
    cd "$SLEPC_SRC"
    /usr/bin/python3 -X faulthandler ./configure
else
    echo ">>> [SKIP] SLEPc already configured"
fi

if ls "$SLEPC_SRC/$PETSC_ARCH/lib/libslepc.so."* 1>/dev/null 2>&1; then
    echo ">>> [SKIP] SLEPc already built"
else
    echo ">>> Building SLEPc"
    cd "$SLEPC_SRC"
    make -j"$NPROC" SLEPC_DIR="$SLEPC_DIR" PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
    checkpoint_done "slepc"
fi

cd "$PROJECT_DIR"

### ------------------------------------------------------------
### 4. mpi4py
### ------------------------------------------------------------

if ! python3 -c "import mpi4py" &>/dev/null; then
    echo ">>> Installing mpi4py"
    pip uninstall -y mpi4py || true
    pip cache purge || true
    MPICC="$MPI_DIR/bin/mpicc" pip install --no-binary=mpi4py mpi4py
else
    echo ">>> [SKIP] mpi4py already installed"
fi

### ------------------------------------------------------------
### 5. petsc4py (desde el propio arbol de PETSc: misma version)
### ------------------------------------------------------------

if ! python3 -c "import petsc4py" &>/dev/null; then
    echo ">>> Installing petsc4py from $PETSC_DIR/src/binding/petsc4py"
    pip uninstall -y petsc petsc4py || true
    pip install --no-deps --no-build-isolation "$PETSC_DIR/src/binding/petsc4py"
else
    echo ">>> [SKIP] petsc4py already installed"
fi

### ------------------------------------------------------------
### 6. slepc4py (desde el propio arbol de SLEPc: misma version)
### ------------------------------------------------------------

if ! python3 -c "import slepc4py" &>/dev/null; then
    echo ">>> Installing slepc4py from $SLEPC_DIR/src/binding/slepc4py"
    pip uninstall -y slepc slepc4py || true
    pip install --no-deps --no-build-isolation "$SLEPC_DIR/src/binding/slepc4py"
else
    echo ">>> [SKIP] slepc4py already installed"
fi

### ------------------------------------------------------------
### 7. Patch activate
### ------------------------------------------------------------

ACTIVATE_FILE="$VENV_DIR/bin/activate"

if ! grep -q "PETSC_DIR" "$ACTIVATE_FILE"; then
    echo ">>> Patching activate script"
    cat << EOF >> "$ACTIVATE_FILE"

# --- NOTA: antes de activar este entorno hay que cargar los modulos
# --- (o usar directamente: source $ENV_FILE)

# --- PETSc configuration (inplace) ---
export PETSC_DIR="$PETSC_SRC"
export PETSC_ARCH="$PETSC_ARCH"

# --- SLEPc configuration (inplace) ---
export SLEPC_DIR="$SLEPC_SRC"

# --- Runtime library paths ---
export LD_LIBRARY_PATH="\$PETSC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="\$SLEPC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"

# --- MUMPS OpenMP threads (ajustar segun el job) ---
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-8}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --- MPI compiler wrapper ---
export MPICC="$MPI_DIR/bin/mpicc"
EOF
else
    echo ">>> [SKIP] activate script already patched"
fi

### ------------------------------------------------------------
### 7.1 Script de entorno (modulos + venv) para sesiones y .slurm
### ------------------------------------------------------------

cat > "$ENV_FILE" << EOF
# Uso:  source $ENV_FILE
module purge
module load apps/avx512-2025
module load $MODULES
source "$VENV_DIR/bin/activate"
EOF
echo ">>> Written $ENV_FILE"

### ------------------------------------------------------------
### 8. Sanity check
### ------------------------------------------------------------

echo ">>> Running sanity test"
python3 - << 'PYEOF'
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import sys

print("PETSc version :", PETSc.Sys.getVersion())
print("SLEPc version :", SLEPc.Sys.getVersion())
print("MPI size      :", MPI.COMM_WORLD.Get_size())
print("Scalar type   :", PETSc.ScalarType)

if not PETSc.Sys.hasExternalPackage("mumps"):
    print("MUMPS         : ERROR - PETSc compilado sin MUMPS")
    sys.exit(1)
print("MUMPS         : OK")
PYEOF

### ------------------------------------------------------------
### 9. Extra Python packages
### ------------------------------------------------------------

pip install netCDF4 scipy matplotlib h5py

echo ""
echo "========================================"
echo " Installation complete!"
echo "========================================"
echo ""
echo " Activar el entorno (sesion interactiva o .slurm):"
echo "   source $ENV_FILE"
echo ""
echo " MUMPS 5.5.1 compilado con:"
echo "   - OpenMP (-fopenmp + BLR_MT)"
echo "   - MPI distribuido"
echo "   - METIS"
echo "   - SCOTCH + PT-Scotch"
echo ""
echo " Job script recomendado:"
echo "   #SBATCH --ntasks=2"
echo "   #SBATCH --cpus-per-task=8"
echo "   #SBATCH --nodes=1"
echo "   export OMP_NUM_THREADS=8"
echo "   Y en Python: opts['icntl_16'] = 8"
echo "========================================"
