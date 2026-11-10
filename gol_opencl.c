/*
 * Game of Life - versao OpenCL (programa HOST)
 *
 * Base: gol.c (serial). Kernels em gol_kernel.cl: "evolve" (3x3 + regra) e
 * "copy_back" (new -> univ), espelhando as duas fases do evolve() serial.
 * NDRange 1D linearizado: 1 work-item por celula; global = multiplo de local.
 *
 * Uso:      ./gol_opencl [largura altura iteracoes [local_size [arquivo_dump]]]
 *           (padrao: 500 500 50000 256, sem dump)
 * Ambiente: GOL_PLATFORM=<idx>  GOL_DEVICE=<idx>  GOL_KERNEL=<caminho .cl>
 *           (padrao: 1a GPU encontrada; kernel "gol_kernel.cl" no diretorio atual)
 * Compilar: gcc -O2 -g -o gol_opencl gol_opencl.c -lOpenCL
 *           (a biblioteca vai DEPOIS do fonte: com --as-needed, o comando do
 *            enunciado "-lOpenCL -o ... .c" pode falhar no link)
 * Codigos de saida: 0 ok | 1 erro | 2 local_size invalido para o dispositivo
 */
#define _POSIX_C_SOURCE 199309L      /* clock_gettime em C99 estrito */
#define CL_TARGET_OPENCL_VERSION 120 /* clCreateCommandQueue (exigida) sem aviso */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

#define CHECK_CL(err, msg) \
    do { if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "ERRO OpenCL: %s (%d)\n", (msg), (int)(err)); \
        exit(EXIT_FAILURE); } } while (0)

/* Extensao NVIDIA (usada so para registrar dados da ocupacao teorica) */
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV      0x4002
#define CL_DEVICE_WARP_SIZE_NV                0x4003
#endif

/* ------------------------------------------------------------------ */
/* Utilitarios                                                         */
/* ------------------------------------------------------------------ */
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* T = END - START (ns) de um cl_event (fila com CL_QUEUE_PROFILING_ENABLE) */
static double event_ns(cl_event ev)
{
    cl_ulong t0, t1;
    cl_int err;
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    CHECK_CL(err, "clGetEventProfilingInfo(START)");
    err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
    CHECK_CL(err, "clGetEventProfilingInfo(END)");
    return (double)(t1 - t0);
}

static char *read_text_file(const char *path, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Nao foi possivel abrir o kernel '%s' (defina GOL_KERNEL)\n", path);
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "Falha ao ler '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    buf[n] = '\0';
    fclose(f);
    *len_out = (size_t)n;
    return buf;
}

/* Le uma string de clGetDeviceInfo/clGetPlatformInfo (alocada no heap) */
static char *dev_string(cl_device_id dev, cl_device_info param)
{
    size_t sz = 0;
    cl_int err = clGetDeviceInfo(dev, param, 0, NULL, &sz);
    CHECK_CL(err, "clGetDeviceInfo(tamanho string)");
    char *s = calloc(sz + 1, 1);
    err = clGetDeviceInfo(dev, param, sz, s, NULL);
    CHECK_CL(err, "clGetDeviceInfo(string)");
    return s;
}

static char *plat_string(cl_platform_id p, cl_platform_info param)
{
    size_t sz = 0;
    cl_int err = clGetPlatformInfo(p, param, 0, NULL, &sz);
    CHECK_CL(err, "clGetPlatformInfo(tamanho string)");
    char *s = calloc(sz + 1, 1);
    err = clGetPlatformInfo(p, param, sz, s, NULL);
    CHECK_CL(err, "clGetPlatformInfo(string)");
    return s;
}

/* ------------------------------------------------------------------ */
/* Plataforma / dispositivo                                            */
/* ------------------------------------------------------------------ */
static void select_device(cl_platform_id *plat_out, cl_device_id *dev_out)
{
    cl_uint np = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &np);
    CHECK_CL(err, "clGetPlatformIDs(contagem)");
    if (np == 0) {
        fprintf(stderr, "Nenhuma plataforma OpenCL (verifique /etc/OpenCL/vendors e o driver)\n");
        exit(EXIT_FAILURE);
    }
    cl_platform_id *plats = malloc(np * sizeof *plats);
    err = clGetPlatformIDs(np, plats, NULL);
    CHECK_CL(err, "clGetPlatformIDs");

    const char *ep = getenv("GOL_PLATFORM"), *ed = getenv("GOL_DEVICE");
    int want_p = ep ? atoi(ep) : -1;
    int want_d = ed ? atoi(ed) : 0;

    printf("Plataformas OpenCL:\n");
    for (cl_uint p = 0; p < np; p++) {
        char *nm = plat_string(plats[p], CL_PLATFORM_NAME);
        printf("  [%u] %s\n", p, nm);
        free(nm);
    }

    /* 1o tenta GPU; se nao houver, aceita qualquer tipo (ex.: CPU) */
    cl_device_type types[2] = { CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL };
    for (int t = 0; t < 2; t++) {
        for (cl_uint p = 0; p < np; p++) {
            if (want_p >= 0 && (cl_uint)want_p != p) continue;
            cl_uint nd = 0;
            err = clGetDeviceIDs(plats[p], types[t], 0, NULL, &nd);
            if (err == CL_DEVICE_NOT_FOUND) continue;       /* esperado */
            CHECK_CL(err, "clGetDeviceIDs(contagem)");
            if (nd == 0 || (cl_uint)want_d >= nd) continue;
            cl_device_id *devs = malloc(nd * sizeof *devs);
            err = clGetDeviceIDs(plats[p], types[t], nd, devs, NULL);
            CHECK_CL(err, "clGetDeviceIDs");
            *plat_out = plats[p];
            *dev_out  = devs[want_d];
            printf("Selecionado: plataforma %u, dispositivo %d (%s)\n", p, want_d,
                   t == 0 ? "GPU" : "fallback: qualquer tipo");
            free(devs);
            free(plats);
            return;
        }
    }
    fprintf(stderr, "Nenhum dispositivo OpenCL encontrado\n");
    exit(EXIT_FAILURE);
}

/* Limites reais do dispositivo (secao 3.4 do enunciado). Devolve max WG
 * e max work-item size[0] para validacao do local size. */
static void print_device_info(cl_device_id dev, size_t *max_wg, size_t *max_wi0)
{
    cl_int err;
    cl_uint cu = 0, dims = 0;
    cl_ulong lmem = 0, gmem = 0;

    char *name = dev_string(dev, CL_DEVICE_NAME);
    char *vendor = dev_string(dev, CL_DEVICE_VENDOR);
    char *dver = dev_string(dev, CL_DEVICE_VERSION);
    char *cver = dev_string(dev, CL_DEVICE_OPENCL_C_VERSION);
    char *ext = dev_string(dev, CL_DEVICE_EXTENSIONS);

    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cu, &cu, NULL);
    CHECK_CL(err, "CL_DEVICE_MAX_COMPUTE_UNITS");
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof *max_wg, max_wg, NULL);
    CHECK_CL(err, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
    err = clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof lmem, &lmem, NULL);
    CHECK_CL(err, "CL_DEVICE_LOCAL_MEM_SIZE");
    err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof gmem, &gmem, NULL);
    CHECK_CL(err, "CL_DEVICE_GLOBAL_MEM_SIZE");
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof dims, &dims, NULL);
    CHECK_CL(err, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    size_t *wis = malloc(dims * sizeof *wis);
    err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, dims * sizeof *wis, wis, NULL);
    CHECK_CL(err, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
    *max_wi0 = wis[0];

    printf("--- Dispositivo ---\n");
    printf("Nome               : %s\n", name);
    printf("Vendor             : %s\n", vendor);
    printf("Versao OpenCL      : %s\n", dver);
    printf("Versao OpenCL C    : %s\n", cver);
    printf("Compute units      : %u\n", cu);
    printf("Max work-group size: %zu\n", *max_wg);
    printf("Max work-item sizes: ");
    for (cl_uint i = 0; i < dims; i++) printf("%zu%s", wis[i], i + 1 < dims ? " x " : "\n");
    printf("Memoria local      : %lu bytes\n", (unsigned long)lmem);
    printf("Memoria global     : %lu bytes\n", (unsigned long)gmem);

    if (strstr(ext, "cl_nv_device_attribute_query")) {
        cl_uint ma = 0, mi = 0, regs = 0, warp = 0;
        err = clGetDeviceInfo(dev, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof ma, &ma, NULL);
        CHECK_CL(err, "CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV");
        err = clGetDeviceInfo(dev, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof mi, &mi, NULL);
        CHECK_CL(err, "CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV");
        err = clGetDeviceInfo(dev, CL_DEVICE_REGISTERS_PER_BLOCK_NV, sizeof regs, &regs, NULL);
        CHECK_CL(err, "CL_DEVICE_REGISTERS_PER_BLOCK_NV");
        err = clGetDeviceInfo(dev, CL_DEVICE_WARP_SIZE_NV, sizeof warp, &warp, NULL);
        CHECK_CL(err, "CL_DEVICE_WARP_SIZE_NV");
        printf("[NVIDIA] compute capability %u.%u | registradores/bloco %u | warp %u\n",
               ma, mi, regs, warp);
    } else {
        printf("[info] extensao cl_nv_device_attribute_query ausente (dispositivo nao-NVIDIA)\n");
    }

    free(wis); free(name); free(vendor); free(dver); free(cver); free(ext);
}

static void write_dump(const cl_uint *grid, int w, int h, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror("Erro ao criar o arquivo de dump"); return; }
    fprintf(f, "%d %d\n", w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) fputc(grid[(size_t)y * w + x] ? '1' : '0', f);
        fputc('\n', f);
    }
    fclose(f);
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    int w = 500, h = 500, max_iter = 50000;
    size_t local = 256;
    const char *dump_path = NULL;
    const char *kernel_path = getenv("GOL_KERNEL");
    if (!kernel_path) kernel_path = "gol_kernel.cl";

    if (argc != 1 && (argc < 4 || argc > 6)) {
        fprintf(stderr, "Uso: %s [largura altura iteracoes [local_size [arquivo_dump]]]\n", argv[0]);
        return 1;
    }
    if (argc >= 4) { w = atoi(argv[1]); h = atoi(argv[2]); max_iter = atoi(argv[3]); }
    if (argc >= 5) local = (size_t)atol(argv[4]);
    if (argc == 6) dump_path = argv[5];
    if (w < 1 || h < 1 || max_iter < 0 || local < 1 || (long long)w * h > (1LL << 30)) {
        fprintf(stderr, "Parametros invalidos\n");
        return 1;
    }

    const size_t n_cells = (size_t)w * (size_t)h;
    const size_t bytes   = n_cells * sizeof(cl_uint);
    const size_t n_gen   = (size_t)max_iter + 1;     /* laco serial: iter <= max_iter */
    cl_int err;

    /* ================= T_total comeca aqui (inclui setup) ================= */
    double t_begin = now_s();

    /* ---- Plataforma, dispositivo, contexto, fila ---- */
    cl_platform_id platform;
    cl_device_id device;
    select_device(&platform, &device);
    size_t dev_max_wg, dev_max_wi0;
    print_device_info(device, &dev_max_wg, &dev_max_wi0);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    /* ---- Programa e kernels ---- */
    size_t src_len;
    char *src = read_text_file(kernel_path, &src_len);
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, &src_len, &err);
    CHECK_CL(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        cl_int e2 = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        CHECK_CL(e2, "clGetProgramBuildInfo(tamanho do log)");
        char *log = calloc(log_sz + 1, 1);
        e2 = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        CHECK_CL(e2, "clGetProgramBuildInfo(log)");
        fprintf(stderr, "--- Build log ---\n%s\n", log);
        CHECK_CL(err, "clBuildProgram");
    }
    cl_kernel k_evolve = clCreateKernel(prog, "evolve", &err);
    CHECK_CL(err, "clCreateKernel(evolve)");
    cl_kernel k_copy = clCreateKernel(prog, "copy_back", &err);
    CHECK_CL(err, "clCreateKernel(copy_back)");

    /* ---- Validacao do local size contra os limites reais ---- */
    size_t kwg_e = 0, kwg_c = 0, pref_mult = 0;
    err = clGetKernelWorkGroupInfo(k_evolve, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof kwg_e, &kwg_e, NULL);
    CHECK_CL(err, "CL_KERNEL_WORK_GROUP_SIZE(evolve)");
    err = clGetKernelWorkGroupInfo(k_copy, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof kwg_c, &kwg_c, NULL);
    CHECK_CL(err, "CL_KERNEL_WORK_GROUP_SIZE(copy_back)");
    err = clGetKernelWorkGroupInfo(k_evolve, device,
                                   CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                   sizeof pref_mult, &pref_mult, NULL);
    CHECK_CL(err, "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
    size_t kwg = kwg_e < kwg_c ? kwg_e : kwg_c;
    printf("Kernel: max work-group (evolve=%zu, copy_back=%zu) | multiplo preferido=%zu\n",
           kwg_e, kwg_c, pref_mult);
    if (local > dev_max_wg || local > dev_max_wi0 || local > kwg) {
        fprintf(stderr, "local_size=%zu invalido: limite do dispositivo=%zu, "
                        "work-item[0]=%zu, limite do kernel=%zu\n",
                local, dev_max_wg, dev_max_wi0, kwg);
        return 2;
    }

    /* ---- Buffers de dispositivo ---- */
    cl_mem d_univ = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer(univ)");
    cl_mem d_new = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer(new)");

    /* ---- Argumentos (fixos: sem ping-pong, nada a trocar por geracao) ---- */
    err  = clSetKernelArg(k_evolve, 0, sizeof(cl_mem), &d_univ);
    err |= clSetKernelArg(k_evolve, 1, sizeof(cl_mem), &d_new);
    err |= clSetKernelArg(k_evolve, 2, sizeof(cl_int), &w);
    err |= clSetKernelArg(k_evolve, 3, sizeof(cl_int), &h);
    CHECK_CL(err, "clSetKernelArg(evolve)");
    err  = clSetKernelArg(k_copy, 0, sizeof(cl_mem), &d_new);
    err |= clSetKernelArg(k_copy, 1, sizeof(cl_mem), &d_univ);
    err |= clSetKernelArg(k_copy, 2, sizeof(cl_int), &w);
    err |= clSetKernelArg(k_copy, 3, sizeof(cl_int), &h);
    CHECK_CL(err, "clSetKernelArg(copy_back)");

    /* ---- Estado inicial no host: mesma cruz do serial ---- */
    cl_uint *h_univ = malloc(bytes);
    if (!h_univ) { fprintf(stderr, "Falha de alocacao no host\n"); return 1; }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            h_univ[(size_t)y * w + x] = (x == w / 2 || y == h / 2) ? 1 : 0;

    double t_setup_end = now_s();               /* fim do setup */

    /* ---- Transferencia Host -> Device ---- */
    cl_event ev_w;
    double tw0 = now_s();
    err = clEnqueueWriteBuffer(queue, d_univ, CL_TRUE, 0, bytes, h_univ, 0, NULL, &ev_w);
    CHECK_CL(err, "clEnqueueWriteBuffer");
    double t_write_host = now_s() - tw0;

    /* ---- Laco de geracoes: evolve + copy_back, 1 evento por lancamento ---- */
    const size_t global = ((n_cells + local - 1) / local) * local;  /* multiplo de local */
    cl_event *ev_evolve = malloc(n_gen * sizeof(cl_event));
    cl_event *ev_copy   = malloc(n_gen * sizeof(cl_event));
    if (!ev_evolve || !ev_copy) { fprintf(stderr, "Falha de alocacao de eventos\n"); return 1; }

    double tl0 = now_s();
    for (size_t it = 0; it < n_gen; it++) {
        err = clEnqueueNDRangeKernel(queue, k_evolve, 1, NULL, &global, &local,
                                     0, NULL, &ev_evolve[it]);
        CHECK_CL(err, "clEnqueueNDRangeKernel(evolve)");
        err = clEnqueueNDRangeKernel(queue, k_copy, 1, NULL, &global, &local,
                                     0, NULL, &ev_copy[it]);
        CHECK_CL(err, "clEnqueueNDRangeKernel(copy_back)");
    }
    err = clFinish(queue);
    CHECK_CL(err, "clFinish(laco)");
    double t_loop_host = now_s() - tl0;

    /* ---- Tempos de kernel por eventos (soma de todas as geracoes) ---- */
    double ns_evolve = 0.0, ns_copy = 0.0;
    for (size_t it = 0; it < n_gen; it++) {
        ns_evolve += event_ns(ev_evolve[it]);
        ns_copy   += event_ns(ev_copy[it]);
        err = clReleaseEvent(ev_evolve[it]); CHECK_CL(err, "clReleaseEvent(evolve)");
        err = clReleaseEvent(ev_copy[it]);   CHECK_CL(err, "clReleaseEvent(copy_back)");
    }
    double ns_kernel = ns_evolve + ns_copy;

    /* ---- Transferencia Device -> Host ---- */
    cl_event ev_r;
    double tr0 = now_s();
    err = clEnqueueReadBuffer(queue, d_univ, CL_TRUE, 0, bytes, h_univ, 0, NULL, &ev_r);
    CHECK_CL(err, "clEnqueueReadBuffer");
    err = clFinish(queue);
    CHECK_CL(err, "clFinish(leitura)");
    double t_read_host = now_s() - tr0;

    double t_end = now_s();                     /* fim do T_total OpenCL */
    /* ================= T_total termina aqui ================= */

    double ns_write = event_ns(ev_w), ns_read = event_ns(ev_r);

    /* ---- Contagem de vivas no host (como no serial; fora do T_total) ---- */
    int alive = 0;
    for (size_t i = 0; i < n_cells; i++) alive += (int)h_univ[i];
    printf("Total de celulas vivas ao final: %d\n", alive);
    if (dump_path) write_dump(h_univ, w, h, dump_path);

    double t_total  = t_end - t_begin;
    double t_setup  = t_setup_end - t_begin;
    double t_kernel = ns_kernel / 1e9;

    printf("--- Configuracao ---\n");
    printf("Grade %dx%d | %zu geracoes | local=%zu global=%zu (%zu grupos)\n",
           w, h, n_gen, local, global, global / local);
    printf("--- Tempos ---\n");
    printf("T_setup (plataforma..buffers) : %.6f s\n", t_setup);
    printf("T_write HtoD (host / evento)  : %.6f s / %.6f s\n", t_write_host, ns_write / 1e9);
    printf("T_laco  (host, enqueue+fim)   : %.6f s\n", t_loop_host);
    printf("T_kernel evolve (soma eventos): %.0f ns (media %.0f ns/lancamento)\n",
           ns_evolve, ns_evolve / (double)n_gen);
    printf("T_kernel copy   (soma eventos): %.0f ns (media %.0f ns/lancamento)\n",
           ns_copy, ns_copy / (double)n_gen);
    printf("T_kernel total  (soma eventos): %.0f ns = %.6f s\n", ns_kernel, t_kernel);
    printf("T_read  DtoH (host / evento)  : %.6f s / %.6f s\n", t_read_host, ns_read / 1e9);
    printf("T_total (host, completo)      : %.6f s\n", t_total);
    printf("Custo transf/setup = T_total - T_kernel: %.6f s\n", t_total - t_kernel);

    /* Linha unica parseavel para scripts */
    printf("RESULT w=%d h=%d iters=%zu local=%zu global=%zu alive=%d "
           "t_setup_s=%.9f t_write_s=%.9f t_loop_s=%.9f "
           "k_evolve_ns=%.0f k_copy_ns=%.0f k_total_ns=%.0f "
           "t_read_s=%.9f t_total_s=%.9f\n",
           w, h, n_gen, local, global, alive, t_setup, t_write_host, t_loop_host,
           ns_evolve, ns_copy, ns_kernel, t_read_host, t_total);

    /* ---- Liberacao ---- */
    err = clReleaseEvent(ev_w); CHECK_CL(err, "clReleaseEvent(write)");
    err = clReleaseEvent(ev_r); CHECK_CL(err, "clReleaseEvent(read)");
    err = clReleaseMemObject(d_univ);   CHECK_CL(err, "clReleaseMemObject(univ)");
    err = clReleaseMemObject(d_new);    CHECK_CL(err, "clReleaseMemObject(new)");
    err = clReleaseKernel(k_evolve);    CHECK_CL(err, "clReleaseKernel(evolve)");
    err = clReleaseKernel(k_copy);      CHECK_CL(err, "clReleaseKernel(copy_back)");
    err = clReleaseProgram(prog);       CHECK_CL(err, "clReleaseProgram");
    err = clReleaseCommandQueue(queue); CHECK_CL(err, "clReleaseCommandQueue");
    err = clReleaseContext(ctx);        CHECK_CL(err, "clReleaseContext");
    free(ev_evolve); free(ev_copy); free(h_univ); free(src);
    return 0;
}
