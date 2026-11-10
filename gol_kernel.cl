/*
 * Game of Life - kernels OpenCL C (DEVICE)  -- versao BASE, sem otimizacoes
 *
 * Espelha o evolve() do gol.c serial:
 *   evolve    : calcula a proxima geracao de "univ" em "nw" (vizinhanca 3x3,
 *               modulo toroidal em x e y, subtracao da propria celula, regra B3/S23)
 *   copy_back : copia "nw" para "univ" (equivale ao 2o laco do evolve serial)
 *
 * Mapeamento: NDRange 1D linearizado, 1 work-item por celula (gid = y*w + x).
 * O global size e arredondado para multiplo do local size, entao work-items com
 * gid >= w*h saem sem fazer nada (so o ultimo work-group tem itens ociosos).
 * Tipo uint (4 bytes) e layout row-major, como no serial.
 */

__kernel void evolve(__global const uint *univ,
                     __global uint       *nw,
                     const int            w,
                     const int            h)
{
    int gid = (int)get_global_id(0);
    if (gid >= w * h) return;

    int y = gid / w;
    int x = gid % w;
    int n = 0;

    for (int y1 = y - 1; y1 <= y + 1; y1++) {
        for (int x1 = x - 1; x1 <= x + 1; x1++) {
            if (univ[((y1 + h) % h) * w + ((x1 + w) % w)]) {
                n++;
            }
        }
    }

    if (univ[gid]) n--;

    nw[gid] = (n == 3 || (n == 2 && univ[gid]));
}

__kernel void copy_back(__global const uint *nw,
                        __global uint       *univ,
                        const int            w,
                        const int            h)
{
    int gid = (int)get_global_id(0);
    if (gid >= w * h) return;

    univ[gid] = nw[gid];
}
