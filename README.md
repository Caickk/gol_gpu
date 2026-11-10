# Game of Life em OpenCL: como executar o `run_all.sh`

O `run_all.sh` compila, executa, mede e gera as tabelas do Trabalho 4 (Programação para GPU, TEC508/PGCC011). Ele funciona em GPU NVIDIA, Intel, AMD ou mesmo em CPU via OpenCL; as partes exclusivas da NVIDIA (`nvidia-smi`) são puladas quando não existem.

## Arquivos necessários (mesmo diretório)

| Arquivo | Função |
|---|---|
| `gol.c` | versão serial (baseline) |
| `gol_opencl.c` | programa host OpenCL |
| `gol_kernel.cl` | kernels (`evolve` e `copy_back`) |
| `run_all.sh` | script de execução |

## Requisitos

- `gcc`, `python3` e `/usr/bin/time` (pacote `time`)
- Header e loader OpenCL: `ocl-icd-opencl-dev`
- Driver OpenCL do seu dispositivo (NVIDIA, Intel NEO, AMD...). Para testar sem GPU: `pocl-opencl-icd`
- Opcionais: `clinfo` e `nvidia-smi`

```bash
chmod +x run_all.sh
```

## Teste rápido (poucos segundos)

Antes de rodar a bateria completa, confirme que tudo funciona com parâmetros pequenos:

```bash
W=64 H=64 ITER=60 REPS=2 LOCAL_LIST="32 64" VAL_CASES="16x16x10 7x5x51" \
SWEEP_SIZES="16 32 64" SWEEP_ITER=20 ./run_all.sh all
```

## Execução completa

Os padrões são grade 500×500 e 50000 iterações (50001 gerações). O baseline serial leva pelo menos 200 s, então rode em segundo plano:

```bash
nohup ./run_all.sh > run_all.out 2>&1 &
tail -f run_all.out
```

## Executar só uma parte

```bash
./run_all.sh 3.2.3              # só /usr/bin/time (seção 3.2.3)
./run_all.sh localsize          # só variação do local size
./run_all.sh serial measure     # várias etapas
./run_all.sh tables             # regera tabelas e README sem executar nada
./run_all.sh help               # mostra a lista de etapas
```

| Etapa | Seção / passo | O que faz |
|---|---|---|
| `env` | 8 / r2 | clinfo, vendors, lscpu, nvidia-smi |
| `build` | 3.1 / r3 | compila serial e OpenCL |
| `serial` | 3.1.1 / r1 | baseline serial |
| `valid` | r4 | compara a grade final com o serial |
| `measure` | 3.2.1, 3.2.2 / r5 | tempo de kernel (eventos) e total (host) |
| `time` | 3.2.3 | `/usr/bin/time -v` |
| `localsize` | 3.3 / r6 | local size de 32 a 512 |
| `occupancy` | 3.4 / r7 | limites do dispositivo e ocupação teórica |
| `opt` | 3.5 / r8 | variantes de kernel (otimizações), se existirem |
| `metrics` | 3.6 / r9 | bandwidth, GIOPS e speedup (usa dados já coletados) |
| `compare` | 3.7 / r10 | varredura de N, crossover e tabela comparativa |
| `monitor` | 3.8 / r11 | `nvidia-smi dmon` (só NVIDIA) |
| `tables` | - | regera tabelas e README |
| `all` | - | todas, na ordem acima |

Cada etapa pode ser indicada pelo nome, pelo número da seção ou por `rN` (número do passo do roteiro).

Dica: `metrics` não executa o programa; ela calcula a partir dos dados de `serial`, `measure` e `localsize`. Rode essas etapas antes.

## Variáveis úteis

Defina antes do comando, por exemplo `REPS=5 ./run_all.sh localsize`.

| Variável | Padrão | Significado |
|---|---|---|
| `W`, `H`, `ITER` | 500, 500, 50000 | grade e iterações do problema base |
| `REPS` | 3 | repetições por configuração (média e desvio) |
| `LOCAL_LIST` | `32 64 128 256 512` | local sizes testados |
| `LOCAL_DEFAULT` | 256 | local size das demais etapas |
| `SWEEP_SIZES` / `SWEEP_ITER` | `32 ... 4096` / 100 | grades e gerações do crossover |
| `GOL_PLATFORM` / `GOL_DEVICE` | primeira GPU | escolhe plataforma e dispositivo OpenCL por índice |
| `OUT` | `./resultados` | diretório de saída |
| `PEAK_BW_GBS`, `PEAK_GOPS` | 672 / 16300 (só TITAN RTX) | picos para os percentuais em outros dispositivos |
| `MAX_THREADS_PER_CU`, `MAX_WG_PER_CU`, `REGS_PER_CU`, `KERNEL_REGS` | vazio | dados para a ocupação teórica em dispositivos não-NVIDIA |
| `OPS_PER_CELL` | 70 | modelo de contagem de operações inteiras (GIOPS) |

Para escolher outro dispositivo, veja os índices impressos pelo programa (ou por `clinfo`) e use, por exemplo, `GOL_PLATFORM=1 GOL_DEVICE=0 ./run_all.sh all`.

## Comparação com OpenMP e MPI

O arquivo `dados_anteriores.conf` é criado na primeira execução. Preencha os tempos (wall-clock, em segundos) e regere as tabelas:

```
OPENMP_WALL_S=123.4
MPI_WALL_S=98.7
```

```bash
./run_all.sh tables
```

Campos vazios aparecem como `⟨preencher⟩` na tabela final. Compare sempre na mesma máquina e com os mesmos parâmetros.

## Onde ficam os resultados

```
resultados/
├── README.md                 relatório geral com todas as tabelas
├── tabelas/                  um .md por seção
├── csv/                      dados para gráficos
├── 00_ambiente ... 11_monitoramento/   saídas brutas e logs de cada etapa
├── bin/                      executáveis compilados
├── parametros.txt            parâmetros da última execução
└── comandos_executados.log   log de tudo que foi executado
```

## Problemas comuns

- **`fatal error: CL/cl.h`**: instale `ocl-icd-opencl-dev` ou use `OCL_CFLAGS=-I/caminho/include`.
- **Nenhuma plataforma OpenCL**: confira `ls /etc/OpenCL/vendors/` (deve ter o `.icd` do seu driver) e rode `clinfo`.
- **`/usr/bin/time` ausente**: instale o pacote `time`.
- **Local size 512 marcado como inválido**: o dispositivo (comum em Intel/AMD) limita o work-group; a tabela registra isso e o script segue.
- **Ocupação com `n/d`**: em dispositivos não-NVIDIA, informe `MAX_THREADS_PER_CU` e as demais variáveis de ocupação.
- **Poucas amostras no `monitor`**: aumente `MON_ITER` e `MON_REPEAT`, pois o `nvidia-smi dmon` amostra a cada 1 s.
