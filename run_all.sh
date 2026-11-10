#!/usr/bin/env bash
###############################################################################
# run_all.sh - TEC508/PGCC011 - Programação para GPU (OpenCL)
# Game of Life: serial (gol.c) e OpenCL (gol_opencl.c + gol_kernel.cl)
#
# Uso:  ./run_all.sh [ETAPA ...]        (sem argumentos = all)
#       ./run_all.sh 3.2.3              só /usr/bin/time (seção 3.2.3)
#       ./run_all.sh localsize metrics  várias etapas
#       ./run_all.sh help               mostra este cabeçalho (etapas e aliases)
#
# ETAPAS (ordem do roteiro do enunciado; aliases = nº da seção ou do passo rN):
#   env       r2        clinfo, /etc/OpenCL/vendors, lscpu, nvidia-smi (seção 8 / 3.8)
#   build     r3 3.1.2  compila serial e OpenCL (3.1)
#   serial    r1 3.1.1  baseline serial, /usr/bin/time -v (3.1.1)
#   valid     r4        corretude OpenCL x serial, grade inteira (3.1.2)
#   measure   r5 3.2.1 3.2.2   T_kernel por eventos + T_total no host (3.2.1/3.2.2)
#   time      3.2.3     /usr/bin/time -v sobre o programa OpenCL (3.2.3)
#   localsize r6 3.3    local size 32..512, REPS execuções cada (3.3)
#   occupancy r7 3.4    clGetDeviceInfo + ocupação teórica (3.4)
#   opt       r8 3.5    variantes de kernel (otimizações), se existirem (3.5)
#   metrics   r9 3.6    bandwidth, GIOPS, speedup (3.6) - só tabelas, usa dados
#                       de serial/measure/localsize
#   compare   r10 3.7   varredura de N (crossover) + tabela comparativa (3.7)
#   monitor   r11 3.8   nvidia-smi dmon durante a execução (3.8)
#   tables              regera todas as tabelas e o README.md
#   all                 tudo, na ordem acima
# Toda execução termina regerando as tabelas das etapas feitas (e das derivadas)
# e o README.md geral com o conteúdo de todas as tabelas existentes.
#
# Saídas (OUT=./resultados): 00_ambiente ... 11_monitoramento (dados brutos e
# logs de cada etapa), tabelas/ (um .md por seção), csv/, README.md, bin/,
# parametros.txt e comandos_executados.log.
###############################################################################
set -uo pipefail

#============================ CONFIGURAÇÃO ====================================
SRC_DIR="${SRC_DIR:-$PWD}"
pick() { for f in "$@"; do [[ -f $f ]] && { readlink -f "$f"; return; }; done; readlink -m "$1"; }
SERIAL_SRC="${SERIAL_SRC:-$(pick "$SRC_DIR/gol.c" "$SRC_DIR/programa_serial.c" "$SRC_DIR/gol_serial.c")}"
OCL_SRC="${OCL_SRC:-$(pick "$SRC_DIR/gol_opencl.c" "$SRC_DIR/programa_opencl.c")}"
KERNEL_SRC="${KERNEL_SRC:-$(pick "$SRC_DIR/gol_kernel.cl" "$SRC_DIR/kernel.cl")}"

W="${W:-500}"; H="${H:-500}"; ITER="${ITER:-50000}"   # problema base (iter <= ITER => ITER+1 gerações)
REPS="${REPS:-3}"                      # repetições (>=3 pelo enunciado) p/ measure, time, localsize, opt, sweep
SERIAL_REPS="${SERIAL_REPS:-1}"        # baseline serial (>=200 s): 1 execução
LOCAL_LIST="${LOCAL_LIST:-32 64 128 256 512}"
LOCAL_DEFAULT="${LOCAL_DEFAULT:-256}"  # local size de measure/time/opt/monitor/sweep
VAL_CASES="${VAL_CASES:-64x64x100 50x37x200 7x5x51 500x500x300}"   # wxhxiter
SWEEP_SIZES="${SWEEP_SIZES:-32 64 128 256 512 1024 2048 4096}"     # grades NxN (crossover)
SWEEP_ITER="${SWEEP_ITER:-100}"
SWEEP_REPS="${SWEEP_REPS:-$REPS}"
MON_ITER="${MON_ITER:-$ITER}"; MON_REPEAT="${MON_REPEAT:-5}"; MON_INTERVAL="${MON_INTERVAL:-1}"
NVSMI_ID="${NVSMI_ID:-0}"              # índice da GPU no nvidia-smi
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"
CC="${CC:-gcc}"; CFLAGS="${CFLAGS:--O2 -g}"
OCL_CFLAGS="${OCL_CFLAGS:-}"           # ex.: -I/opt/opencl/include
OCL_LIBS="${OCL_LIBS:--lOpenCL}"       # vai DEPOIS do fonte (ld --as-needed)
OPS_PER_CELL="${OPS_PER_CELL:-70}"     # modelo de contagem de operações inteiras (ver README)
PEAK_BW_GBS="${PEAK_BW_GBS:-}"         # vazio: 672 se TITAN RTX, senão n/d
PEAK_GOPS="${PEAK_GOPS:-}"             # vazio: 16300 se TITAN RTX, senão n/d
MAX_THREADS_PER_CU="${MAX_THREADS_PER_CU:-}"; MAX_WG_PER_CU="${MAX_WG_PER_CU:-}"
REGS_PER_CU="${REGS_PER_CU:-}"; LOCAL_MEM_PER_CU="${LOCAL_MEM_PER_CU:-}"
WARP_SIZE="${WARP_SIZE:-}"; KERNEL_REGS="${KERNEL_REGS:-}"; KERNEL_LOCAL_BYTES="${KERNEL_LOCAL_BYTES:-0}"
OPT_VARIANTS="${OPT_VARIANTS:-baseline=$KERNEL_SRC coalescencia=$SRC_DIR/gol_kernel_coalesc.cl local=$SRC_DIR/gol_kernel_local.cl constant=$SRC_DIR/gol_kernel_constant.cl}"
PREV_CONF="${PREV_CONF:-$SRC_DIR/dados_anteriores.conf}"

OUT="${OUT:-$PWD/resultados}"
BIN="$OUT/bin"; TAB="$OUT/tabelas"; CSV="$OUT/csv"
D_ENV="$OUT/00_ambiente"; D_BUILD="$OUT/01_build"; D_SER="$OUT/02_serial"; D_VAL="$OUT/03_validacao"
D_MEAS="$OUT/04_medicao"; D_TIME="$OUT/05_time"; D_LS="$OUT/06_localsize"; D_OCC="$OUT/07_ocupacao"
D_OPT="$OUT/08_otimizacoes"; D_CMP="$OUT/10_comparacao"; D_MON="$OUT/11_monitoramento"
SERIAL_BIN="$BIN/programa_serial"; OCL_BIN="$BIN/programa_opencl"
LOG="$OUT/comandos_executados.log"
export GOL_KERNEL="$KERNEL_SRC"        # lido pelo host OpenCL

#============================ ETAPAS E ALIASES ================================
ORDER=(env build serial valid measure time localsize occupancy opt metrics compare monitor tables)
canon() { case "$1" in
  all) echo all;;
  env|r2) echo env;;
  build|r3|3.1.2) echo build;;
  serial|r1|3.1.1) echo serial;;
  3.1) echo "build serial valid";;
  valid|r4) echo valid;;
  measure|r5|3.2.1|3.2.2) echo measure;;
  3.2) echo "measure time";;
  time|3.2.3) echo time;;
  localsize|r6|3.3) echo localsize;;
  occupancy|r7|3.4) echo occupancy;;
  opt|r8|3.5) echo opt;;
  metrics|r9|3.6|3.6.1|3.6.2|3.6.3) echo metrics;;
  compare|r10|3.7) echo compare;;
  monitor|r11) echo monitor;;
  3.8) echo "env monitor";;
  tables) echo tables;;
  help|-h|--help) echo help;;
  *) echo "?$1";;
esac; }
STEPS=()
for a in "$@"; do for s in $(canon "$a"); do STEPS+=("$s"); done; done
[[ ${#STEPS[@]} -eq 0 ]] && STEPS=(all)
for s in "${STEPS[@]}"; do
  if [[ $s == help ]]; then sed -n '2,/^###*$/p' "$0" | sed 's/^# \{0,1\}//' | head -45; exit 0; fi
  [[ $s == \?* ]] && { echo "Etapa desconhecida: ${s#?} (use ./run_all.sh help)" >&2; exit 1; }
done
want() { [[ " ${STEPS[*]} " == *" all "* || " ${STEPS[*]} " == *" $1 "* ]]; }

mkdir -p "$BIN" "$TAB" "$CSV" "$D_ENV" "$D_BUILD" "$D_SER" "$D_VAL" "$D_MEAS" "$D_TIME" "$D_LS" "$D_OCC" "$D_OPT" "$D_CMP" "$D_MON"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }

# parâmetros desta execução (lidos pelo analyze.py); "tables" sozinho preserva os anteriores
if ! { [[ ${#STEPS[@]} -eq 1 && ${STEPS[0]} == tables ]] && [[ -f "$OUT/parametros.txt" ]]; }; then
  { for v in W H ITER REPS SERIAL_REPS LOCAL_LIST LOCAL_DEFAULT VAL_CASES SWEEP_SIZES SWEEP_ITER SWEEP_REPS \
             MON_ITER MON_REPEAT MON_INTERVAL OPS_PER_CELL PEAK_BW_GBS PEAK_GOPS MAX_THREADS_PER_CU MAX_WG_PER_CU \
             REGS_PER_CU LOCAL_MEM_PER_CU WARP_SIZE KERNEL_REGS KERNEL_LOCAL_BYTES OPT_VARIANTS PREV_CONF \
             SERIAL_SRC OCL_SRC KERNEL_SRC; do printf '%s="%s"\n' "$v" "${!v}"; done; } > "$OUT/parametros.txt"
fi
if [[ ! -f "$PREV_CONF" ]]; then
  cat > "$PREV_CONF" <<'EOF'
# Resultados dos trabalhos anteriores (tabela comparativa 3.7). Preencha em segundos
# (wall-clock) e rode ./run_all.sh compare (ou tables). Vazio = placeholder na tabela.
OPENMP_WALL_S=
OPENMP_NOTE=
MPI_WALL_S=
MPI_NOTE=
EOF
fi

#=========================== FUNÇÕES AUXILIARES ===============================
need_time() { [[ -x /usr/bin/time ]] || { echo "ERRO: /usr/bin/time não existe (instale o pacote 'time')." >&2; exit 1; }; }

do_build() {
  log "### build"
  [[ -f "$SERIAL_SRC" && -f "$OCL_SRC" && -f "$KERNEL_SRC" ]] || { log "ERRO: fonte não encontrado (SERIAL_SRC=$SERIAL_SRC OCL_SRC=$OCL_SRC KERNEL_SRC=$KERNEL_SRC)"; exit 1; }
  local c1="$CC $CFLAGS -o $SERIAL_BIN $SERIAL_SRC"
  local c2="$CC $CFLAGS $OCL_CFLAGS -o $OCL_BIN $OCL_SRC $OCL_LIBS"
  echo "$c1" > "$D_BUILD/serial.cmd"; echo "$c2" > "$D_BUILD/opencl.cmd"
  log "$c1"; $c1 > "$D_BUILD/serial.log" 2>&1; echo $? > "$D_BUILD/serial.rc"
  log "$c2"; $c2 > "$D_BUILD/opencl.log" 2>&1; echo $? > "$D_BUILD/opencl.rc"
  if [[ $(cat "$D_BUILD/serial.rc") != 0 || $(cat "$D_BUILD/opencl.rc") != 0 ]]; then
    log "ERRO de compilação (veja $D_BUILD/*.log). Dica: instale ocl-icd-opencl-dev (header CL/cl.h) ou use OCL_CFLAGS/OCL_LIBS."
    return 1
  fi
}
ensure_bins() {
  if [[ ! -x $SERIAL_BIN || ! -x $OCL_BIN || $SERIAL_SRC -nt $SERIAL_BIN || $OCL_SRC -nt $OCL_BIN ]]; then
    do_build || exit 1
  fi
}

# ocl_run BASE w h iter local [dump]  (TIMED=1 => /usr/bin/time -v). Gera BASE.out/.err/.rc[/.time]
ocl_run() {
  local base=$1 rc; shift
  log "RUN opencl: $* -> ${base#$OUT/}"
  if [[ ${TIMED:-0} == 1 ]]; then
    /usr/bin/time -v -o "$base.time" timeout "$RUN_TIMEOUT" "$OCL_BIN" "$@" < /dev/null > "$base.out" 2> "$base.err"; rc=$?
  else
    timeout "$RUN_TIMEOUT" "$OCL_BIN" "$@" < /dev/null > "$base.out" 2> "$base.err"; rc=$?
  fi
  echo "$rc" > "$base.rc"
  case $rc in 0) ;; 2) log "  local size inválido para o dispositivo (rc=2)";; *) log "  AVISO: retorno $rc (veja ${base#$OUT/}.err)";; esac
  return 0
}
# serial_run BASE w h iter [dump]  (TIMED=1 => time -v)
serial_run() {
  local base=$1 rc; shift
  log "RUN serial: $* -> ${base#$OUT/}"
  if [[ ${TIMED:-0} == 1 ]]; then
    /usr/bin/time -v -o "$base.time" timeout "$RUN_TIMEOUT" "$SERIAL_BIN" "$@" < /dev/null > "$base.out" 2> "$base.err"; rc=$?
  else
    timeout "$RUN_TIMEOUT" "$SERIAL_BIN" "$@" < /dev/null > "$base.out" 2> "$base.err"; rc=$?
  fi
  echo "$rc" > "$base.rc"; (( rc != 0 )) && log "  AVISO: retorno $rc (veja ${base#$OUT/}.err)"
  return 0
}

ANALYZE=()

#================================ ETAPAS ======================================
# ---- env: ambiente (seção 8, roteiro 2) --------------------------------------
if want env; then
  log "### env"; ensure_bins
  { ls -la /etc/OpenCL/vendors/ 2>&1; for f in /etc/OpenCL/vendors/*.icd; do [[ -f $f ]] && echo "$f: $(cat "$f")"; done; } > "$D_ENV/vendors.txt" 2>&1
  if command -v clinfo >/dev/null 2>&1; then clinfo > "$D_ENV/clinfo.txt" 2>&1; else echo "clinfo não encontrado" > "$D_ENV/clinfo.txt"; log "AVISO: clinfo não instalado."; fi
  { lscpu; } > "$D_ENV/lscpu.txt" 2>&1
  { uname -a; grep PRETTY_NAME /etc/os-release; $CC --version | head -1; free -h; } > "$D_ENV/sistema.txt" 2>&1
  { ls -l /usr/include/CL/cl.h 2>&1; ls -l /usr/lib/x86_64-linux-gnu/libOpenCL.so* 2>&1; } > "$D_ENV/opencl_dev.txt" 2>&1
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$D_ENV/nvidia-smi.txt" 2>&1; nvidia-smi -q > "$D_ENV/nvidia-smi-q.txt" 2>&1
  else echo "nvidia-smi não encontrado (dispositivo não-NVIDIA ou driver ausente)" > "$D_ENV/nvidia-smi.txt"; fi
  ocl_run "$D_ENV/devinfo_host" 32 32 0 32     # limites reais via clGetDeviceInfo (programa host)
  ANALYZE+=(env)
fi

# ---- build (3.1) ---------------------------------------------------------------
if want build; then do_build || exit 1; ANALYZE+=(build); fi

# ---- serial (3.1.1, roteiro 1) ---------------------------------------------------
if want serial; then
  need_time; ensure_bins
  log "### serial: ${W}x${H}, iter=$ITER (SERIAL_REPS=$SERIAL_REPS)"
  for i in $(seq 1 "$SERIAL_REPS"); do TIMED=1 serial_run "$D_SER/serial_r$i" "$W" "$H" "$ITER"; done
  ANALYZE+=(serial)
fi

# ---- valid (3.1.2, roteiro 4): grade inteira, serial x OpenCL --------------------
if want valid; then
  log "### valid: casos {$VAL_CASES} x local {$LOCAL_LIST}"; ensure_bins
  for c in $VAL_CASES; do
    IFS=x read -r vw vh vi <<< "$c"
    sb="$D_VAL/serial_$c"; serial_run "$sb" "$vw" "$vh" "$vi" "$sb.dump"
    for L in $LOCAL_LIST; do
      ob="$D_VAL/ocl_${c}_L$L"; ocl_run "$ob" "$vw" "$vh" "$vi" "$L" "$ob.dump"
      if [[ $(cat "$ob.rc") == 0 && -f "$sb.dump" && -f "$ob.dump" ]]; then
        if cmp -s "$sb.dump" "$ob.dump"; then echo PASS > "$ob.cmp"; else echo FAIL > "$ob.cmp"; fi
      else echo NA > "$ob.cmp"; fi
    done
  done
  ANALYZE+=(valid)
fi

# ---- measure (3.2.1 / 3.2.2, roteiro 5): eventos + T_total no host -----------------
if want measure; then
  log "### measure: ${W}x${H} iter=$ITER local=$LOCAL_DEFAULT REPS=$REPS"; ensure_bins
  for i in $(seq 1 "$REPS"); do ocl_run "$D_MEAS/run_r$i" "$W" "$H" "$ITER" "$LOCAL_DEFAULT"; done
  ANALYZE+=(measure)
fi

# ---- time (3.2.3): /usr/bin/time -v sobre o programa OpenCL -------------------------
if want time; then
  need_time; ensure_bins
  log "### time (3.2.3): ${W}x${H} iter=$ITER local=$LOCAL_DEFAULT REPS=$REPS"
  for i in $(seq 1 "$REPS"); do TIMED=1 ocl_run "$D_TIME/run_r$i" "$W" "$H" "$ITER" "$LOCAL_DEFAULT"; done
  ANALYZE+=(time)
fi

# ---- localsize (3.3, roteiro 6) -------------------------------------------------------
if want localsize; then
  log "### localsize: {$LOCAL_LIST} x REPS=$REPS (${W}x${H}, iter=$ITER)"; ensure_bins
  for L in $LOCAL_LIST; do
    for i in $(seq 1 "$REPS"); do
      ocl_run "$D_LS/L${L}_r$i" "$W" "$H" "$ITER" "$L"
      [[ $(cat "$D_LS/L${L}_r$i.rc") == 2 ]] && break      # inválido: não repete
    done
  done
  ANALYZE+=(localsize)
fi

# ---- occupancy (3.4, roteiro 7): limites reais do dispositivo -------------------------
if want occupancy; then
  log "### occupancy"; ensure_bins
  ocl_run "$D_OCC/devinfo" 32 32 0 32
  command -v clinfo >/dev/null 2>&1 && clinfo > "$D_OCC/clinfo.txt" 2>&1
  ANALYZE+=(occupancy)
fi

# ---- opt (3.5, roteiro 8): variantes de kernel, se existirem ------------------------
if want opt; then
  log "### opt: variantes {$OPT_VARIANTS}"; ensure_bins
  serial_run "$D_OPT/serial_val" 64 64 100 "$D_OPT/serial_val.dump"
  for v in $OPT_VARIANTS; do
    name=${v%%=*}; file=${v#*=}
    if [[ ! -f $file ]]; then echo "ausente: $file" > "$D_OPT/$name.status"; log "variante '$name': $file não existe (placeholder na tabela)"; continue; fi
    echo "presente: $file" > "$D_OPT/$name.status"
    ( export GOL_KERNEL="$file"
      ocl_run "$D_OPT/${name}_val" 64 64 100 "$LOCAL_DEFAULT" "$D_OPT/${name}_val.dump"
      if [[ $(cat "$D_OPT/${name}_val.rc") == 0 ]] && cmp -s "$D_OPT/serial_val.dump" "$D_OPT/${name}_val.dump"; then echo PASS > "$D_OPT/${name}_val.cmp"; else echo FAIL > "$D_OPT/${name}_val.cmp"; fi
      for i in $(seq 1 "$REPS"); do ocl_run "$D_OPT/${name}_r$i" "$W" "$H" "$ITER" "$LOCAL_DEFAULT"; done )
  done
  ANALYZE+=(opt)
fi

# ---- metrics (3.6, roteiro 9): só tabelas ---------------------------------------------
if want metrics; then
  ANALYZE+=(metrics)
  [[ -n "$(ls "$D_LS"/*.out "$D_MEAS"/*.out 2>/dev/null)" ]] || log "AVISO: sem dados de measure/localsize; rode essas etapas (e serial) antes de metrics."
fi

# ---- compare (3.7, roteiro 10): varredura de N (crossover) + tabela final ---------------
if want compare; then
  log "### compare: N={$SWEEP_SIZES} iter=$SWEEP_ITER, REPS=$SWEEP_REPS, local=$LOCAL_DEFAULT"; ensure_bins
  for n in $SWEEP_SIZES; do
    serial_run "$D_CMP/serial_N$n" "$n" "$n" "$SWEEP_ITER"
    for i in $(seq 1 "$SWEEP_REPS"); do ocl_run "$D_CMP/ocl_N${n}_r$i" "$n" "$n" "$SWEEP_ITER" "$LOCAL_DEFAULT"; done
  done
  ANALYZE+=(compare)
fi

# ---- monitor (3.8, roteiro 11): nvidia-smi dmon ------------------------------------------
if want monitor; then
  log "### monitor (nvidia-smi dmon)"; ensure_bins
  rm -f "$D_MON/indisponivel.txt"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi não encontrado: monitoramento indisponível neste dispositivo (não-NVIDIA ou driver ausente)." > "$D_MON/indisponivel.txt"
    log "AVISO: nvidia-smi ausente; etapa registrada como indisponível."
  else
    nvidia-smi > "$D_MON/nvidia-smi_antes.txt" 2>&1
    nvidia-smi -q -d CLOCK > "$D_MON/nvidia-smi_clock.txt" 2>&1
    nvidia-smi dmon -i "$NVSMI_ID" -s pucm -d "$MON_INTERVAL" > "$D_MON/dmon.txt" 2> "$D_MON/dmon.err" &
    DMON=$!; sleep 2
    ( sleep 3; nvidia-smi > "$D_MON/nvidia-smi_durante.txt" 2>&1 ) &
    SNAP=$!
    for i in $(seq 1 "$MON_REPEAT"); do ocl_run "$D_MON/run_r$i" "$W" "$H" "$MON_ITER" "$LOCAL_DEFAULT"; done
    sleep 2; wait "$SNAP" 2>/dev/null; kill "$DMON" 2>/dev/null; wait "$DMON" 2>/dev/null
    nvidia-smi > "$D_MON/nvidia-smi_depois.txt" 2>&1
  fi
  ANALYZE+=(monitor)
fi

#============================ TABELAS E README ================================
log "### tabelas"
cat > "$OUT/analyze.py" <<'PYEOF'
# -*- coding: utf-8 -*-
import os, re, sys, glob, math, datetime, statistics as st
OUT=os.environ["OUT"]; TAB=OUT+"/tabelas"; CSVD=OUT+"/csv"; NAN=float("nan")
def D(n): return OUT+"/"+n
P={}
for l in open(D("parametros.txt"),errors="ignore"):
    if "=" in l: k,v=l.rstrip("\n").split("=",1); P[k]=v.strip('"')
def pnum(k,default=NAN):
    try: return float(P.get(k,"").strip())
    except ValueError: return default
LOCALS=[int(x) for x in P.get("LOCAL_LIST","32 64 128 256 512").split()]
LDEF=int(P.get("LOCAL_DEFAULT","256") or 256); OPS=pnum("OPS_PER_CELL",70.0)

# ---------------------------------------------------------------- utilitários
def mean(x):
    x=[v for v in x if v==v]; return sum(x)/len(x) if x else NAN
def sd(x):
    x=[v for v in x if v==v]; return st.stdev(x) if len(x)>1 else (0.0 if x else NAN)
def fmt(x,d=3): return "n/d" if x!=x else f"{x:.{d}f}"
def ms(x,d=3,sc=1.0):
    m=mean(x); return "n/d" if m!=m else f"{m*sc:.{d}f} ± {sd(x)*sc:.{d}f}"
def md(h,rows):
    s="| "+" | ".join(h)+" |\n|"+"|".join(["---"]*len(h))+"|\n"
    return s+"".join("| "+" | ".join(str(c) for c in r)+" |\n" for r in rows)
def save(name,title,body):
    t=f"## {title}\n\n{body}\n"; open(f"{TAB}/{name}.md","w").write(t); print(t)
def read(p):
    try: return open(p,errors="ignore").read()
    except OSError: return ""
def rc_of(base):
    t=read(base+".rc").strip(); return int(t) if t.lstrip("-").isdigit() else None
def time_of(p):
    d={}
    for l in read(p).splitlines():
        try:
            if "Elapsed (wall clock)" in l:
                v=[float(x) for x in l.strip().rsplit(" ",1)[1].split(":")]; d["wall"]=sum(x*60**i for i,x in enumerate(reversed(v)))
            elif "User time" in l: d["user"]=float(l.rsplit(" ",1)[1])
            elif "System time" in l: d["sys"]=float(l.rsplit(" ",1)[1])
            elif "Maximum resident" in l: d["rss"]=float(l.rsplit(" ",1)[1])/1024
        except Exception: pass
    return d
def parse_result(base):
    m=re.search(r"^RESULT (.*)$",read(base+".out"),re.M)
    if not m: return None
    r={}
    for kv in m.group(1).split():
        k,v=kv.split("=",1); r[k]=float(v)
    r["time"]=time_of(base+".time"); r["base"]=base; return r
def runs_of(d,prefix):
    out=[]
    for p in sorted(glob.glob(f"{d}/{prefix}_r*.out"),key=lambda z:int(re.search(r"_r(\d+)\.out$",z)[1])):
        b=p[:-4]
        if rc_of(b)==0:
            r=parse_result(b)
            if r: out.append(r)
    return out
def serial_out(base):
    t=read(base+".out"); a=re.search(r"vivas ao final:\s*(\d+)",t); s=re.search(r"Tempo interno de execucao:\s*(\S+)",t)
    return (float(s[1]) if s else NAN, int(a[1]) if a else None)

# métricas de um RESULT (bytes/ns = GB/s)
def kst(r):
    N=r["w"]*r["h"]; g=r["iters"]; ke,kc,kt=r["k_evolve_ns"],r["k_copy_ns"],r["k_total_ns"]
    return dict(bw_comp=g*N*16/kt, bw_log=g*N*48/kt, ev_comp=g*N*8/ke, ev_log=g*N*40/ke, cp=g*N*8/kc,
                gi_ev=g*N*OPS/ke, gi_tot=g*N*OPS/kt, kt=kt, tt=r["t_total_s"])

# ----------------------------------------------------------- dispositivo (host)
def host_dev(t):
    g=lambda p:(re.search(p,t,re.M) or [None,None])[1]
    d=dict(nome=g(r"^Nome\s*:\s*(.*)$"),vendor=g(r"^Vendor\s*:\s*(.*)$"),versao=g(r"^Versao OpenCL\s*:\s*(.*)$"),
           versao_c=g(r"^Versao OpenCL C\s*:\s*(.*)$"),cu=g(r"^Compute units\s*:\s*(\d+)"),maxwg=g(r"^Max work-group size:\s*(\d+)"),
           maxwi=g(r"^Max work-item sizes:\s*(.*)$"),lmem=g(r"^Memoria local\s*:\s*(\d+)"),gmem=g(r"^Memoria global\s*:\s*(\d+)"))
    m=re.search(r"compute capability (\d+)\.(\d+) \| registradores/bloco (\d+) \| warp (\d+)",t)
    if m: d.update(cc=(int(m[1]),int(m[2])),regs_blk=int(m[3]),warp=int(m[4]))
    m=re.search(r"evolve=(\d+), copy_back=(\d+)\) \| multiplo preferido=(\d+)",t)
    if m: d.update(kwg=min(int(m[1]),int(m[2])),pref=int(m[3]))
    return d if d["nome"] else None
def any_dev():
    for p in [D("07_ocupacao/devinfo.out"),D("00_ambiente/devinfo_host.out")]+sorted(glob.glob(D("04_medicao/*.out"))+glob.glob(D("06_localsize/*.out"))):
        d=host_dev(read(p))
        if d: return d
    return None
def peaks(dev):
    bw,go=pnum("PEAK_BW_GBS"),pnum("PEAK_GOPS")
    if dev and "TITAN RTX" in (dev["nome"] or ""):
        if bw!=bw: bw=672.0
        if go!=go: go=16300.0
    return bw,go
def pct(x,peak): return "n/d" if (x!=x or peak!=peak) else f"{100*x/peak:.2f}%"

NV_CC={(7,5):(1024,16,65536,65536),(7,0):(2048,32,65536,98304),(8,0):(2048,32,65536,167936),
       (8,6):(1536,16,65536,102400),(8,9):(1536,24,65536,102400),(6,1):(2048,32,65536,98304)}

# =================================================================== SEÇÕES
def env():
    d=D("00_ambiente"); cl=read(d+"/clinfo.txt"); hd=host_dev(read(d+"/devinfo_host.out")) or {}
    def c(p): m=re.search(p,cl,re.M); return m[1].strip() if m else "n/d"
    h=lambda k,suf="":(f"{hd[k]}{suf}" if hd.get(k) else "n/d")
    rows=[["Plataforma",c(r"^\s*Platform Name\s+(.*)$"),"(listada no log do host)"],
          ["Nome do dispositivo",c(r"^\s*Device Name\s+(.*)$"),h("nome")],
          ["Versão OpenCL",c(r"^\s*Device Version\s+(.*)$"),h("versao")],
          ["Max compute units",c(r"^\s*Max compute units\s+(\d+)"),h("cu")],
          ["Max work-group size",c(r"^\s*Max work group size\s+(\d+)"),h("maxwg")],
          ["Memória local (B)",c(r"^\s*Local memory size\s+(\d+)"),h("lmem")],
          ["Memória global (B)",c(r"^\s*Global memory size\s+(\d+)"),h("gmem")]]
    b="### Dispositivo OpenCL (clinfo x clGetDeviceInfo no programa host)\n\n"+md(["campo","clinfo","programa host"],rows)
    nm=(hd.get("nome") or "")+c(r"^\s*Device Name\s+(.*)$"); ver=(hd.get("versao") or "")+c(r"^\s*Device Version\s+(.*)$")
    ck=lambda x:"✔" if x else "✘"
    chk=[["Dispositivo é NVIDIA TITAN RTX",ck("TITAN RTX" in nm),"ambiente de referência do enunciado"],
         ["OpenCL 3.0 suportado",ck("3.0" in ver),hd.get("versao") or c(r"^\s*Device Version\s+(.*)$")],
         ["nvidia.icd em /etc/OpenCL/vendors",ck("nvidia.icd" in read(d+"/vendors.txt")),"ICD loader"],
         ["clinfo disponível",ck("não encontrado" not in cl and cl.strip()!=""),""],
         ["nvidia-smi disponível",ck("não encontrado" not in read(d+"/nvidia-smi.txt") and read(d+"/nvidia-smi.txt").strip()!=""),"monitoramento 3.8 só em NVIDIA"]]
    b+="\n### Verificações do ambiente (seção 8)\n\n"+md(["item","status","observação"],chk)
    b+="\n_✘ em item NVIDIA não invalida a execução: o script segue em outros dispositivos OpenCL (Intel/AMD/CPU)._\n"
    m=lambda p:(re.search(p,read(d+"/lscpu.txt"),re.M) or [None,"n/d"])[1].strip()
    b+="\n### Processador (lscpu) e sistema\n\n"+md(["campo","valor"],[["Modelo",m(r"^Model name:\s*(.*)$")],["Núcleos lógicos",m(r"^CPU\(s\):\s*(.*)$")],
        ["Threads/núcleo",m(r"^Thread\(s\) per core:\s*(.*)$")],["Sistema",(read(d+"/sistema.txt").splitlines() or ["n/d"])[0]]])
    b+="\nSaídas completas: `00_ambiente/clinfo.txt`, `vendors.txt`, `lscpu.txt`, `nvidia-smi.txt`, `devinfo_host.out`.\n"
    save("00_ambiente","Ambiente (seção 8 / roteiro 2)",b)

def build():
    rows=[]
    for n in ("serial","opencl"):
        cmd=read(D(f"01_build/{n}.cmd")).strip()
        if not cmd: continue
        rc=read(D(f"01_build/{n}.rc")).strip(); lg=read(D(f"01_build/{n}.log"))
        rows.append([n,f"`{cmd}`","OK" if rc=="0" else f"FALHOU (rc={rc})",len(re.findall("warning",lg))])
    if rows: save("01_build","Compilação (3.1)",md(["versão","comando","status","warnings"],rows)+
        "\n> O enunciado mostra `gcc -O2 -g -lOpenCL -o ... .c`; em distribuições com `--as-needed` a biblioteca precisa vir **depois** do fonte, como acima.\n")

def serial():
    rows=[];ts=[]
    for p in sorted(glob.glob(D("02_serial/serial_r*.out"))):
        b=p[:-4]; t,a=serial_out(b); tm=time_of(b+".time")
        if t==t: ts.append(t); rows.append([os.path.basename(b),fmt(t),fmt(tm.get("wall",NAN)),fmt(tm.get("user",NAN)),fmt(tm.get("sys",NAN)),fmt(tm.get("rss",NAN),1),a])
    if not rows: return
    b=md(["execução","tempo interno (s)","wall (time -v)","user (s)","system (s)","Max RSS (MB)","células vivas"],rows)
    b+=f"\n**T_serial médio = {mean(ts):.3f} s (desvio {sd(ts):.3f}; {len(ts)} execução(ões))**\n"
    if mean(ts)<200: b+="\n> ⚠ T_serial < 200 s: o enunciado exige baseline ≥ 200 s com -O2 (aumente a grade ou as iterações).\n"
    save("02_serial","Baseline C serial (3.1.1 / roteiro 1)",b)

def valid():
    rows=[];ok=0;tot=0
    for sp in sorted(glob.glob(D("03_validacao/serial_*.out"))):
        c=os.path.basename(sp)[7:-4]; _,sa=serial_out(sp[:-4])
        for L in LOCALS:
            ob=D(f"03_validacao/ocl_{c}_L{L}"); rc=rc_of(ob)
            if rc is None: continue
            if rc==2: rows.append([c,L,sa,"-","L inválido neste dispositivo"]); continue
            oa=(re.search(r"vivas ao final:\s*(\d+)",read(ob+".out")) or [None,None])[1]
            v=read(ob+".cmp").strip() or "NA"; tot+=1; ok+=(v=="PASS")
            rows.append([c,L,sa,oa,"✔ PASS (grade idêntica)" if v=="PASS" else ("✘ FAIL" if v=="FAIL" else "n/d")])
    if rows: save("03_validacao","Validação contra o serial (3.1.2 / roteiro 4)",md(["caso (w x h x iter)","local size","vivas serial","vivas OpenCL","comparação (cmp da grade final)"],rows)+
        f"\n**{ok}/{tot} comparações idênticas.** Problema inteiro: comparação exata (a tolerância de 1e-6 do enunciado não se aplica).\n")

def meas_table(runs,title_sfx=""):
    r0=runs[0]; N=int(r0["w"]*r0["h"])
    cfg=f"Grade {int(r0['w'])}x{int(r0['h'])}, {int(r0['iters'])} gerações, local={int(r0['local'])}, global={int(r0['global'])} ({int(r0['global']/r0['local'])} grupos)\n\n"
    rows=[]
    for i,r in enumerate(runs,1):
        rows.append([i,fmt(r["t_setup_s"],4),fmt(r["t_write_s"]*1e3,3),fmt(r["t_loop_s"],4),f"{r['k_evolve_ns']:.0f}",f"{r['k_copy_ns']:.0f}",
                     f"{r['k_total_ns']:.0f}",fmt(r["t_read_s"]*1e3,3),fmt(r["t_total_s"],4),fmt(r["t_total_s"]-r["k_total_ns"]/1e9,4)])
    g=lambda k,s=1.0:[r[k]*s for r in runs]
    rows.append(["**média ± dp**",ms(g("t_setup_s"),4),ms(g("t_write_s"),3,1e3),ms(g("t_loop_s"),4),ms(g("k_evolve_ns"),0),ms(g("k_copy_ns"),0),ms(g("k_total_ns"),0),
                 ms(g("t_read_s"),3,1e3),ms(g("t_total_s"),4),ms([r["t_total_s"]-r["k_total_ns"]/1e9 for r in runs],4)])
    return cfg,md(["exec","T_setup (s)","T_write HtoD (ms)","T_laço host (s)","T_kernel evolve (ns)","T_kernel copy (ns)","T_kernel total (ns)","T_read DtoH (ms)","T_total (s)","T_total − T_kernel (s)"],rows)

def measure():
    runs=runs_of(D("04_medicao"),"run")
    if not runs: return
    cfg,t=meas_table(runs)
    b=cfg+"### T_kernel (eventos OpenCL) e T_total (clock_gettime no host)\n\n"+t
    b+="\nT_kernel = soma de (END − START) de todos os lançamentos (`evolve` + `copy_back`); T_total = do início do setup ao fim da leitura (inclui contexto, build do kernel, HtoD, laço, DtoH).\n"
    mk=lambda k:mean([r[k] for r in runs]); tt=mk("t_total_s"); kt=mk("k_total_ns")/1e9
    parts=[("Setup (plataforma, contexto, build, buffers)",mk("t_setup_s")),("Transferência HtoD",mk("t_write_s")),("Kernel (soma dos eventos)",kt),
           ("Overhead de lançamento/filas (T_laço − T_kernel)",mk("t_loop_s")-kt),("Transferência DtoH",mk("t_read_s"))]
    rest=tt-sum(v for _,v in parts); parts.append(("Outros (contagem de tempo fora das fases)",rest))
    b+="\n### Classificação dos custos (média, kernel x transferência x setup)\n\n"+md(["componente","tempo (s)","% do T_total"],[[n,fmt(v,5),fmt(100*v/tt,2)] for n,v in parts])
    b+=f"\n**Custo de transferências + setup = T_total − T_kernel = {tt-kt:.4f} s ({100*(tt-kt)/tt:.1f}% do T_total).** \nEm iGPU não há PCIe (memória compartilhada): `HtoD/DtoH` refletem cópia em memória do sistema.\n"
    save("04_medicao","Medição de tempo: eventos OpenCL e host (3.2.1 / 3.2.2 / roteiro 5)",b)

def timex():
    runs=runs_of(D("05_time"),"run")
    if not runs: return
    rows=[]
    for i,r in enumerate(runs,1):
        t=r["time"]; rows.append([i,fmt(r["t_total_s"],4),fmt(r["k_total_ns"]/1e9,4),fmt(t.get("wall",NAN),3),fmt(t.get("user",NAN),2),fmt(t.get("sys",NAN),2),fmt(t.get("rss",NAN),1),fmt(t.get("wall",NAN)-r["t_total_s"],3)])
    g=lambda f:[f(r) for r in runs]
    rows.append(["**média ± dp**",ms(g(lambda r:r["t_total_s"]),4),ms(g(lambda r:r["k_total_ns"]/1e9),4),ms(g(lambda r:r["time"].get("wall",NAN)),3),
                 ms(g(lambda r:r["time"].get("user",NAN)),2),ms(g(lambda r:r["time"].get("sys",NAN)),2),ms(g(lambda r:r["time"].get("rss",NAN)),1),
                 ms(g(lambda r:r["time"].get("wall",NAN)-r["t_total_s"]),3)])
    r0=runs[0]
    b=f"Grade {int(r0['w'])}x{int(r0['h'])}, {int(r0['iters'])} gerações, local={int(r0['local'])}\n\n"+md(
        ["exec","T_total host (s)","T_kernel eventos (s)","wall time -v (s)","user (s)","system (s)","Max RSS (MB)","wall − T_total (s)"],rows)
    b+=("\n> **Leitura:** o `wall` do /usr/bin/time cobre o processo inteiro (carga do driver/ICD, exit); por isso supera o T_total medido dentro do programa. "
        "`user`/`system` incluem as threads do runtime/driver (e o spin-wait do `clFinish`); Max RSS é memória do **host** (a memória do dispositivo não aparece). "
        "O tempo de kernel só é obtido pelos eventos OpenCL.\n")
    save("05_time","Medição externa com /usr/bin/time -v (3.2.3)",b)

def ls_data():
    data={}
    for L in LOCALS:
        rs=runs_of(D("06_localsize"),f"L{L}")
        inval=any(read(f).strip()=="2" for f in glob.glob(D(f"06_localsize/L{L}_r*.rc")))
        data[L]=(rs,inval)
    return data
def localsize():
    data=ls_data(); rows=[];csv="local,global,kernel_ns,kernel_sd,total_ms,total_sd,bw_gbs\n";best=None;bt=None
    for L in LOCALS:
        rs,inv=data[L]
        if inv: rows.append([L,"-","-","inválido (excede o limite do dispositivo/kernel)","-","-"]); continue
        if not rs: rows.append([L,"-","-","(sem dados)","-","-"]); continue
        kt=[r["k_total_ns"] for r in rs]; tt=[r["t_total_s"]*1e3 for r in rs]; bw=[kst(r)["bw_comp"] for r in rs]; bl=[kst(r)["bw_log"] for r in rs]
        rows.append([L,f"{int(rs[0]['global'])} ({int(rs[0]['global']/L)} grupos)",ms(kt,0),ms(tt,2),ms(bw,2),ms(bl,2)])
        csv+=f"{L},{int(rs[0]['global'])},{mean(kt):.0f},{sd(kt):.0f},{mean(tt):.3f},{sd(tt):.3f},{mean(bw):.3f}\n"
        if best is None or mean(kt)<best[1]: best=(L,mean(kt))
        if bt is None or mean(tt)<bt[1]: bt=(L,mean(tt))
    if not any(r[2]!="-" for r in rows): return
    open(CSVD+"/localsize.csv","w").write(csv)
    r0=next(rs[0] for rs,_ in data.values() if rs)
    b=f"Grade {int(r0['w'])}x{int(r0['h'])}, {int(r0['iters'])} gerações; global = menor múltiplo do local ≥ N células. Média ± desvio de {len(next(rs for rs,_ in data.values() if rs))} execuções.\n\n"
    b+=md(["local size","NDRange global","T_kernel (ns)","T_total (ms)","BW efetiva compulsória (GB/s)","BW efetiva lógica (GB/s)"],rows)
    if best: b+=f"\n**Menor T_kernel: local={best[0]} ({best[1]:.0f} ns). Menor T_total: local={bt[0]} ({bt[1]:.1f} ms).**\n"
    dev=any_dev()
    if dev: b+=f"\nContexto: {dev['cu']} compute units, max work-group {dev['maxwg']} (ver tabela de ocupação 3.4).\n"
    b+=("\n**Discussão:** relacionar o melhor local size à ocupação (3.4) e ao nº de compute units: poucos work-items por grupo subutilizam a CU (limite de blocos residentes); "
        "grupos muito grandes reduzem a granularidade de escalonamento. O T_total varia pouco com o local size porque é dominado por setup e overhead de lançamento.\n")
    save("06_localsize","Variação do local size (3.3 / roteiro 6)",b)

def occupancy():
    dev=host_dev(read(D("07_ocupacao/devinfo.out")))
    if not dev: return
    b=md(["parâmetro (clGetDeviceInfo)","valor"],[["Nome",dev["nome"]],["Vendor",dev["vendor"]],["Versão OpenCL",dev["versao"]],["CL_DEVICE_MAX_COMPUTE_UNITS",dev["cu"]],
        ["CL_DEVICE_MAX_WORK_GROUP_SIZE",dev["maxwg"]],["CL_DEVICE_MAX_WORK_ITEM_SIZES",dev["maxwi"]],["CL_DEVICE_LOCAL_MEM_SIZE (B)",dev["lmem"]],
        ["CL_DEVICE_GLOBAL_MEM_SIZE (B)",dev["gmem"]],["CL_KERNEL_WORK_GROUP_SIZE (kernel)",dev.get("kwg","n/d")],["Múltiplo preferido de work-group",dev.get("pref","n/d")]])
    if dev.get("cc"): b+=f"\nNVIDIA: compute capability {dev['cc'][0]}.{dev['cc'][1]}, registradores/bloco {dev['regs_blk']}, warp {dev['warp']}.\n"
    peak=""
    if "TITAN RTX" in (dev["nome"] or ""): peak="Valores esperados no enunciado para a TITAN RTX: 72 SMs, 4608 CUDA cores, 24 GB, 672 GB/s, 16,3 TFLOPS FP32.\n"
    b+=("\n"+peak)
    ref=NV_CC.get(dev.get("cc"))
    def pv(k,i):
        v=pnum(k); return int(v) if v==v else (ref[i] if ref else None)
    maxT,maxB,regs,lmem=pv("MAX_THREADS_PER_CU",0),pv("MAX_WG_PER_CU",1),pv("REGS_PER_CU",2),pv("LOCAL_MEM_PER_CU",3)
    warp=int(pnum("WARP_SIZE")) if pnum("WARP_SIZE")==pnum("WARP_SIZE") else dev.get("warp",dev.get("pref",32))
    R=pnum("KERNEL_REGS"); KL=pnum("KERNEL_LOCAL_BYTES",0.0)
    b+="\n### Parâmetros de ocupação adotados\n\n"+md(["parâmetro","valor","origem"],[
        ["work-items residentes máx. por CU",maxT if maxT else "n/d","env MAX_THREADS_PER_CU ou tabela NVIDIA por compute capability (conferir no CUDA Programming Guide)"],
        ["work-groups residentes máx. por CU",maxB if maxB else "n/d","idem (MAX_WG_PER_CU)"],
        ["registradores por CU",regs if regs else "n/d","idem (REGS_PER_CU)"],
        ["memória local por CU (B)",lmem if lmem else "n/d","idem (LOCAL_MEM_PER_CU)"],
        ["tamanho de warp/wavefront",warp,"extensão NV, WARP_SIZE ou múltiplo preferido do kernel"],
        ["registradores por work-item (kernel)",fmt(R,0) if R==R else "n/d (não consultável via OpenCL padrão; informe KERNEL_REGS, ex.: do PTX/ptxas -v)","KERNEL_REGS"],
        ["memória local por work-group (B)",fmt(KL,0)+(" (kernel base não usa __local)" if KL==0 else ""),"KERNEL_LOCAL_BYTES"]])
    N=int(P.get("W","500"))*int(P.get("H","500")); cu=int(dev["cu"] or 0); rows=[]
    lim=min(int(dev["maxwg"] or 0),dev.get("kwg",10**9))
    for L in LOCALS:
        grp=math.ceil(N/L)
        if L>lim: rows.append([L,grp,"-","-","-","-","-","inválido (excede o limite)"]); continue
        if not maxT: rows.append([L,grp,fmt(grp/cu if cu else NAN,1),"n/d","n/d","n/d","n/d","n/d (informe MAX_THREADS_PER_CU)"]); continue
        w_thr=maxT//L; w_blk=maxB or 10**9
        w_reg=int(regs//(R*math.ceil(L/warp)*warp)) if (regs and R==R and R>0) else 10**9
        w_lm=int(lmem//KL) if (lmem and KL>0) else 10**9
        wg=min(w_thr,w_blk,w_reg,w_lm); lims={"work-items/CU":w_thr,"work-groups/CU":w_blk,"registradores":w_reg,"memória local":w_lm}
        who=", ".join(k for k,v in lims.items() if v==wg and v<10**9)
        rows.append([L,grp,fmt(grp/cu if cu else NAN,1),wg,wg*L,f"{100*wg*L/maxT:.1f}%",f"{100*wg*L/warp/(maxT/warp):.1f}%",who])
    b+="\n### Ocupação teórica por local size\n\n"+md(["local size","grupos no NDRange","grupos por CU","work-groups ativos/CU","work-items ativos/CU","ocupação teórica","(warps ativos/máx.)","fator limitante"],rows)
    b+=("\nOcupação teórica = work-items ativos por CU / máximo suportado, limitada por: work-items/CU, work-groups/CU, registradores e memória local. "
        "O kernel base **não usa `__local`**, então a memória local não limita; um kernel com tiling reduz o nº de grupos residentes conforme `KERNEL_LOCAL_BYTES`. "
        "Sem `KERNEL_REGS` o limite por registradores não é considerado (ocupação pode estar superestimada).\n")
    save("07_ocupacao","Ocupação teórica (3.4 / roteiro 7)",b)

def opt():
    names=[v.split("=",1)[0] for v in P.get("OPT_VARIANTS","").split() if "=" in v]
    if not names or not glob.glob(D("08_otimizacoes/*.status")): return
    base=None;rows=[]
    desc={"baseline":"kernel ingênuo (sem otimizações)","coalescencia":"3.5.1 coalescência","local":"3.5.2 __local + barrier","constant":"3.5.3 __constant"}
    for n in names:
        stt=read(D(f"08_otimizacoes/{n}.status")).strip()
        if not stt: continue
        if stt.startswith("ausente"): rows.append([n,desc.get(n,""),"não implementada","-","-","-","-","-"]); continue
        rs=runs_of(D("08_otimizacoes"),n); v=read(D(f"08_otimizacoes/{n}_val.cmp")).strip()
        if not rs: rows.append([n,desc.get(n,""),"sem dados","✔" if v=="PASS" else "✘","-","-","-","-"]); continue
        kt=mean([r["k_total_ns"] for r in rs]); bw=[kst(r)["bw_comp"] for r in rs]
        if n=="baseline": base=kt
        rows.append([n,desc.get(n,""),"implementada","✔ PASS" if v=="PASS" else "✘ FAIL",ms([r["k_total_ns"] for r in rs],0),
                     fmt(base/kt,2) if base else "n/d",ms(bw,2),ms([r["t_total_s"]*1e3 for r in rs],1)])
    if rows: save("08_otimizacoes","Otimizações de memória (3.5 / roteiro 8)",md(
        ["variante","descrição","status","validação vs serial","T_kernel (ns)","speedup de kernel vs baseline","BW efetiva comp. (GB/s)","T_total (ms)"],rows)+
        "\nO script procura `gol_kernel_coalesc.cl`, `gol_kernel_local.cl` e `gol_kernel_constant.cl` (ou `OPT_VARIANTS=\"nome=arquivo ...\"`). "
        "Cada variante deve manter os kernels `evolve` e `copy_back` com os mesmos argumentos do host (variantes com argumentos extras exigem estender o host). "
        "Exigência do enunciado: pelo menos 2 otimizações com impacto mensurado.\n")

def tserial():
    ts=[]
    for p in sorted(glob.glob(D("02_serial/serial_r*.out"))):
        t,_=serial_out(p[:-4])
        if t==t: ts.append(t)
    return ts
def best_data():
    d=ls_data(); out={L:rs for L,(rs,i) in d.items() if rs}
    if not out:
        m=runs_of(D("04_medicao"),"run")
        if m: out={int(m[0]["local"]):m}
    return out
def metrics():
    data=best_data()
    if not data: return
    dev=any_dev(); pbw,pgo=peaks(dev); ts=mean(tserial())
    r0=next(iter(data.values()))[0]; N=int(r0["w"]*r0["h"]); g=int(r0["iters"])
    rows=[];rows2=[];rows3=[];csv="local,bw_comp,bw_log,giops_ev,speedup_total,speedup_kernel\n"
    for L,rs in sorted(data.items()):
        k=[kst(r) for r in rs]; G=lambda key:[x[key] for x in k]
        rows.append([L,ms(G("ev_comp"),2),ms(G("ev_log"),2),ms(G("cp"),2),ms(G("bw_comp"),2),pct(mean(G("bw_comp")),pbw),ms(G("bw_log"),2),pct(mean(G("bw_log")),pbw)])
        rows2.append([L,ms(G("gi_ev"),2),pct(mean(G("gi_ev")),pgo),ms(G("gi_tot"),2),pct(mean(G("gi_tot")),pgo)])
        kt=[r["k_total_ns"]/1e9 for r in rs]; tt=[r["t_total_s"] for r in rs]; sl=[r["t_total_s"]-r["t_setup_s"] for r in rs]
        rows3.append([L,fmt(ts,3),ms(tt,4),ms(kt,4),fmt(ts/mean(tt),2),fmt(ts/mean(kt),2),fmt(ts/mean(sl),2)])
        csv+=f"{L},{mean(G('bw_comp')):.3f},{mean(G('bw_log')):.3f},{mean(G('gi_ev')):.3f},{ts/mean(tt):.3f},{ts/mean(kt):.3f}\n"
    open(CSVD+"/metricas.csv","w").write(csv)
    pk=lambda v,u:("n/d (informe PEAK_*)" if v!=v else f"{v:g} {u}")
    b=f"Grade {int(r0['w'])}x{int(r0['h'])}, {g} gerações, N={N:,} células. Picos de referência: BW={pk(pbw,'GB/s')}; INT/FP32={pk(pgo,'GOPS')}.\n\n"
    b+=("### 3.6.1 Largura de banda efetiva = (bytes lidos + escritos na memória global) / T_kernel\n\n"
        "Duas definições (reportadas lado a lado): **compulsória** = 1 leitura + 1 escrita de `uint` por célula (evolve 8 B/célula; copy 8 B/célula) e "
        "**lógica** = 9 leituras da vizinhança + 1 escrita (evolve 40 B/célula; as leituras repetidas são absorvidas pelo cache, então a lógica pode superar o pico). "
        "As colunas `total` usam T_kernel = evolve + copy.\n\n")
    b+=md(["local","evolve comp. (GB/s)","evolve lógica (GB/s)","copy (GB/s)","total comp. (GB/s)","% do pico (comp.)","total lógica (GB/s)","% do pico (lógica)"],rows)
    b+=("\n### 3.6.2 Desempenho aritmético (GIOPS = operações inteiras / T_kernel)\n\n"
        f"O Game of Life não usa ponto flutuante: reporta-se **GIOPS**. Modelo de contagem por célula/geração = **{OPS:g} ops** (por vizinho: 2 somas de wrap, 2 módulos, 1 multiplicação, 1 soma de índice, 1 incremento condicional = 7 × 9 = 63; "
        "+1 decremento, +4 da regra B3/S23, +2 de `gid/w` e `gid%w`); `copy_back` não tem aritmética. O pico de referência do enunciado (16,3 T) é de FP32; "
        "módulo/divisão inteiros custam várias instruções, logo a comparação é apenas indicativa. Ajuste com `OPS_PER_CELL`.\n\n")
    b+=md(["local","GIOPS (T_evolve)","% do pico","GIOPS (T_kernel total)","% do pico"],rows2)
    b+=("\n### 3.6.3 Speedup\n\n"
        "Speedup = T_serial / T_total_opencl; Speedup_kernel = T_serial / T_kernel (sem transferências nem setup); `sem setup` = T_serial / (T_total − T_setup). "
        f"T_serial = tempo interno médio do serial ({fmt(ts,3)} s; {len(tserial())} execução(ões)).\n\n")
    b+=md(["local","T_serial (s)","T_total OpenCL (s)","T_kernel (s)","Speedup","Speedup_kernel","Speedup sem setup"],rows3)
    b+=("\n**Diferença Speedup x Speedup_kernel:** o segundo ignora setup (plataforma, contexto, build do kernel), transferências e overhead de lançamento; "
        "é o ganho do kernel isolado. O primeiro é o ganho percebido pelo usuário e é limitado pelo custo fixo, que não escala com o nº de gerações.\n")
    save("09_metricas","Métricas de desempenho (3.6 / roteiro 9)",b)

def sweep():
    rows=[]
    for n in [int(x) for x in P.get("SWEEP_SIZES","").split()]:
        sb=D(f"10_comparacao/serial_N{n}"); st_,sa=serial_out(sb)
        rs=runs_of(D("10_comparacao"),f"ocl_N{n}")
        if st_!=st_ or not rs: continue
        oa=(re.search(r"vivas ao final:\s*(\d+)",read(rs[0]["base"]+".out")) or [None,None])[1]
        rows.append(dict(n=n,ts=st_,tt=mean([r["t_total_s"] for r in rs]),ts_=mean([r["t_total_s"]-r["t_setup_s"] for r in rs]),
                         tk=mean([r["k_total_ns"]/1e9 for r in rs]),ok=(str(sa)==str(oa))))
    return rows
def cross(rows,key):
    c=None
    for r in reversed(rows):
        if r["ts"]>r[key]: c=r["n"]
        else: break
    return c
def readconf():
    c={}
    for l in read(P.get("PREV_CONF","")).splitlines():
        if "=" in l and not l.strip().startswith("#"):
            k,v=l.split("=",1); c[k.strip()]=v.strip()
    return c
def compare():
    conf=readconf(); sw=sweep(); ts=tserial()
    sw_wall=mean([time_of(b[:-4]+".time").get("wall",NAN) for b in glob.glob(D("02_serial/serial_r*.out"))])
    tw=runs_of(D("05_time"),"run"); tm=runs_of(D("04_medicao"),"run")
    if tw: ow=mean([r["time"].get("wall",NAN) for r in tw]); osrc="wall do /usr/bin/time (etapa 3.2.3)"; ttot=mean([r["t_total_s"] for r in tw]); kt=mean([r["k_total_ns"]/1e9 for r in tw])
    elif tm: ow=mean([r["t_total_s"] for r in tm]); osrc="T_total medido no host (3.2.2)"; ttot=ow; kt=mean([r["k_total_ns"]/1e9 for r in tm])
    else: ow=ttot=kt=NAN; osrc="sem dados (rode measure/time)"
    sref=sw_wall if (sw_wall==sw_wall and sw_wall>0) else mean(ts)
    def val(k):
        try: return float(conf.get(k) or P.get(k,"") or "nan")
        except ValueError: return NAN
    ph="⟨preencher⟩"
    def row(nm,w,note):
        return [nm,fmt(w,3) if w==w else ph,("1.00" if nm.startswith("C serial") and w==w else (fmt(sref/w,2) if (w==w and w>0 and sref==sref) else ph)),note]
    rows=[row("C serial (Trabalho 2)",sref,"baseline desta execução (/usr/bin/time -v); a coluna de speedup usa esta linha como T_serial"),
          row("OpenMP, 4 threads (Trabalho 2)",val("OPENMP_WALL_S"),conf.get("OPENMP_NOTE") or "dado do trabalho anterior (dados_anteriores.conf)"),
          row("MPI, 4 processos (Trabalho 3)",val("MPI_WALL_S"),conf.get("MPI_NOTE") or "dado do trabalho anterior (dados_anteriores.conf)"),
          row("OpenCL/GPU (este trabalho)",ow,f"T_total: {osrc}; inclui setup, transferências e kernel"),
          row("OpenCL/GPU, só kernel",kt,"T_kernel (eventos) - ignora setup e transferências")]
    b="### Tabela comparativa final\n\n"+md(["versão","wall-clock time (s)","speedup vs C serial","observações"],rows)
    b+="\n> Compare versões na **mesma máquina** e com os mesmos parâmetros (grade e iterações); preencha `dados_anteriores.conf` com os tempos do OpenMP e do MPI e rode `./run_all.sh tables`.\n"
    if sw:
        trows=[[r["n"],f"{r['n']*r['n']:,}",fmt(r["ts"],4),fmt(r["tt"],4),fmt(r["ts_"],4),fmt(r["tk"],5),fmt(r["ts"]/r["tt"],2),fmt(r["ts"]/r["ts_"],2),
                "GPU" if r["ts"]>r["tt"] else "CPU","✔" if r["ok"] else "✘"] for r in sw]
        c1,c2=cross(sw,"tt"),cross(sw,"ts_")
        b+=f"\n### Crossover point (N x N, {P.get('SWEEP_ITER')} gerações, local={LDEF})\n\n"+md(
            ["N","células","T_serial (s)","T_total GPU (s)","T_total GPU sem setup (s)","T_kernel (s)","speedup total","speedup sem setup","mais rápido (total)","vivas = serial"],trows)
        f=lambda c:f"N = {c}" if c else "não encontrado no intervalo testado"
        b+=f"\n**Crossover (T_total, com setup): {f(c1)}.  Crossover sem o setup fixo: {f(c2)}.**\n"
        b+=("\n**Discussão:** para N pequeno o custo fixo (descoberta de plataforma, contexto, build do kernel, filas, 2 lançamentos por geração) supera o tempo de computação, "
            "e o serial vence; a GPU só compensa quando o trabalho por geração amortiza esses custos. O crossover depende do nº de gerações (aqui "
            f"{P.get('SWEEP_ITER')}): mais gerações deslocam o ponto para N menores. A GPU favorece algoritmos embaraçosamente paralelos e limitados por banda (como este, sem dependência entre células "
            "dentro de uma geração); sofre com dependências seriais longas e pouco paralelismo (a dependência entre gerações aqui é resolvida por lançamentos sequenciais).\n")
        csv="N,serial_s,gpu_total_s,gpu_sem_setup_s,gpu_kernel_s\n"+"".join(f"{r['n']},{r['ts']:.6f},{r['tt']:.6f},{r['ts_']:.6f},{r['tk']:.6f}\n" for r in sw)
        open(CSVD+"/crossover.csv","w").write(csv)
    else: b+="\n### Crossover point\n\n_Sem dados da varredura de N: rode `./run_all.sh compare`._\n"
    save("10_comparacao","Comparação com trabalhos anteriores e crossover (3.7 / roteiro 10)",b)

def monitor():
    d=D("11_monitoramento")
    if os.path.exists(d+"/indisponivel.txt"):
        save("11_monitoramento","Monitoramento (3.8 / roteiro 11)",md(["item","resultado"],[["nvidia-smi dmon","n/d"],["motivo",read(d+"/indisponivel.txt").strip()]])+
             "\n`clinfo` foi coletado na etapa `env`. A utilização/clocks/energia exigem `nvidia-smi` (NVIDIA).\n"); return
    cols=None;rows=[]
    for l in read(d+"/dmon.txt").splitlines():
        if l.startswith("#"):
            t=l[1:].split()
            if "gpu" in t and ("sm" in t or "pwr" in t): cols=t
            continue
        v=l.split()
        if cols and len(v)==len(cols):
            try: rows.append({c:(float(x) if x!="-" else NAN) for c,x in zip(cols,v)})
            except ValueError: pass
    if not rows: return
    act=[r for r in rows if r.get("sm",0)>0] or rows
    g=lambda s,k:[r.get(k,NAN) for r in s]
    mx=lambda x:max([v for v in x if v==v],default=NAN)
    q=read(d+"/nvidia-smi_clock.txt"); m=re.search(r"Max Clocks\s+Graphics\s+:\s+(\d+)",q); maxclk=float(m[1]) if m else NAN
    ds=read(d+"/nvidia-smi_durante.txt"); mm=re.search(r"(\d+)MiB\s*/\s*(\d+)MiB",ds)
    tbl=[["amostras (total / com GPU ativa sm>0)",f"{len(rows)} / {len([r for r in rows if r.get('sm',0)>0])}"],
         ["utilização média da GPU - sm (%) (amostras ativas)",fmt(mean(g(act,'sm')),1)],["utilização média de memória - mem (%)",fmt(mean(g(act,'mem')),1)],
         ["potência média / máxima (W)",f"{fmt(mean(g(act,'pwr')),1)} / {fmt(mx(g(rows,'pwr')),1)}"],
         ["clock gráfico pclk médio / máximo (MHz)",f"{fmt(mean(g(act,'pclk')),0)} / {fmt(mx(g(rows,'pclk')),0)}"],
         ["clock de memória mclk médio / máximo (MHz)",f"{fmt(mean(g(act,'mclk')),0)} / {fmt(mx(g(rows,'mclk')),0)}"],
         ["temperatura máxima (°C)",fmt(mx(g(rows,'gtemp')),0)],
         ["memória usada (fb, MiB) média / máxima",f"{fmt(mean(g(act,'fb')),0)} / {fmt(mx(g(rows,'fb')),0)}"],
         ["memória em nvidia-smi durante a execução",f"{mm[1]} MiB / {mm[2]} MiB" if mm else "n/d"],
         ["clock gráfico máximo do dispositivo (MHz)",fmt(maxclk,0)]]
    b=md(["métrica (nvidia-smi dmon, amostragem a cada "+P.get("MON_INTERVAL","1")+" s)","valor"],tbl)
    pm=mx(g(rows,"pclk"))
    b+=("\n**Boost:** "+("n/d" if (pm!=pm or maxclk!=maxclk) else (f"clock máximo observado {pm:.0f} MHz vs máximo do dispositivo {maxclk:.0f} MHz ({100*pm/maxclk:.0f}%) - "+("atingiu o boost." if pm>=0.95*maxclk else "não atingiu o clock máximo (kernel curto, limite de potência/térmico ou clock ocioso)."))) +"\n")
    b+=(f"\nExecução monitorada: {P.get('MON_REPEAT')} execuções consecutivas, {P.get('MON_ITER')} iterações. Com amostragem de 1 s e kernels curtos, poucas amostras caem dentro do kernel: aumente `MON_ITER`/`MON_REPEAT`.\n"
        "Arquivos: `11_monitoramento/dmon.txt`, `nvidia-smi_antes/durante/depois.txt`.\n")
    save("11_monitoramento","Monitoramento com nvidia-smi dmon (3.8 / roteiro 11)",b)

SECTIONS=[("00_ambiente","Ambiente (seção 8 / roteiro 2)","env"),("01_build","Compilação (3.1)","build"),("02_serial","Baseline C serial (3.1.1 / roteiro 1)","serial"),
 ("03_validacao","Validação contra o serial (3.1.2 / roteiro 4)","valid"),("04_medicao","Medição: eventos OpenCL e host (3.2.1 / 3.2.2 / roteiro 5)","measure"),
 ("05_time","Medição externa com /usr/bin/time -v (3.2.3)","time"),("06_localsize","Variação do local size (3.3 / roteiro 6)","localsize"),
 ("07_ocupacao","Ocupação teórica (3.4 / roteiro 7)","occupancy"),("08_otimizacoes","Otimizações de memória (3.5 / roteiro 8)","opt"),
 ("09_metricas","Métricas de desempenho (3.6 / roteiro 9)","metrics"),("10_comparacao","Comparação com trabalhos anteriores e crossover (3.7 / roteiro 10)","compare"),
 ("11_monitoramento","Monitoramento com nvidia-smi dmon (3.8 / roteiro 11)","monitor")]
def readme():
    dev=any_dev(); s="# Resultados - Game of Life em OpenCL (TEC508/PGCC011, Trabalho 4)\n\n"
    s+=f"Gerado em {datetime.datetime.now():%Y-%m-%d %H:%M:%S}. Dispositivo: {dev['nome'] if dev else 'n/d'}.\n\n"
    s+="## Parâmetros\n\n"+md(["parâmetro","valor"],[[k,f"`{P.get(k,'')}`"] for k in ("W","H","ITER","REPS","SERIAL_REPS","LOCAL_LIST","LOCAL_DEFAULT","SWEEP_SIZES","SWEEP_ITER","OPS_PER_CELL")])
    s+="\n## Índice\n\n"+"".join(f"- {t}\n" for _,t,_ in SECTIONS)+"\n"
    for f,t,step in SECTIONS:
        p=f"{TAB}/{f}.md"
        s+=(read(p).rstrip()+"\n\n") if os.path.exists(p) else f"## {t}\n\n_Etapa não executada ainda: `./run_all.sh {step}`._\n\n"
    open(OUT+"/README.md","w").write(s); print("README.md gerado")

FN=dict(env=env,build=build,serial=serial,valid=valid,measure=measure,time=timex,localsize=localsize,occupancy=occupancy,opt=opt,metrics=metrics,compare=compare,monitor=monitor)
if __name__=="__main__":
    args=sys.argv[1:]
    if "all" in args: args=list(FN)
    for a in args:
        try: FN[a]()
        except Exception as e: print(f"[{a}] erro: {e!r}")
    readme()
PYEOF

# seções derivadas: refresh de metrics/compare quando os dados-fonte mudaram
has() { local x; for x in "${ANALYZE[@]:-}"; do [[ $x == "$1" ]] && return 0; done; return 1; }
if want tables || want all; then ANALYZE=(all)
else
  { has serial || has measure || has localsize; } && ! has metrics && ANALYZE+=(metrics)
  { has serial || has measure || has time || has compare; } && ! has compare && ANALYZE+=(compare)
fi
OUT="$OUT" python3 "$OUT/analyze.py" "${ANALYZE[@]:-}" > "$OUT/analise_stdout.txt" 2>&1
log "Tabelas: $OUT/tabelas/*.md | README geral: $OUT/README.md | CSV: $OUT/csv/"
log "FIM"
