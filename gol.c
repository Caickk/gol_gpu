#define _POSIX_C_SOURCE 199309L   /* clock_gettime em C99 estrito */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

/*
 * ALTERACOES MINIMAS em relacao ao gol.c original (logica de evolucao intacta):
 *   1) w, h, max_iter e arquivo de dump opcional via argv (padrao: 500 500 50000);
 *   2) matrizes "univ" e "new" em heap (antes eram VLAs na pilha), necessario
 *      para grades grandes no estudo de crossover. "new" e alocada UMA vez em
 *      game() e passada a evolve() (mesmo criterio do golmpi.c);
 *   3) dump opcional do estado final para comparacao com a versao OpenCL.
 *
 * Uso:    ./gol [largura altura iteracoes [arquivo_dump]]
 * Compilar: gcc -O2 -g -o gol gol.c
 */

/*
// Função para gerar o arquivo PBM
void save_pbm(void *u, int w, int h, int iter)
{
   int (*univ)[w] = u;
   char filename[64];
   
   // Formata o nome do arquivo para gol_nn.pbm (ex: gol_0, gol_500, gol_1000)
   sprintf(filename, "gol_%d.pbm", iter);
   
   FILE *f = fopen(filename, "w");
   if (!f) {
      perror("Erro ao criar o arquivo PBM");
      return;
   }

   // Cabeçalho e dados PBM no formato P1 (ASCII monocromático)
   fprintf(f, "P1\n%d %d\n", w, h);
   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         fprintf(f, "%d ", univ[y][x]);
      }
      fprintf(f, "\n");
   }
   fclose(f);
}
*/

// Função principal que calcula a próxima geração do Game of Life.
// "nw" e o buffer auxiliar (equivalente ao antigo "unsigned new[h][w]").
void evolve(void *u, void *nw, int w, int h)
{
   unsigned (*univ)[w] = u;
   unsigned (*new)[w] = nw;

   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         int n = 0;
         
         for (int y1 = y - 1; y1 <= y + 1; y1++) {
            for (int x1 = x - 1; x1 <= x + 1; x1++) {
               if (univ[(y1 + h) % h][(x1 + w) % w]) {
                  n++;
               }
            }
         }

         if (univ[y][x]) n--;
         
         new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
      }
   }

   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         univ[y][x] = new[y][x];
      }
   }
}

int count_alive(void *u, int w, int h)
{
    unsigned (*univ)[w] = u;
    int total_alive = 0;
    // Leitura estritamente sequencial, linha por coluna (sem paralelismo)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            total_alive += univ[y][x];
        }
    }
    return total_alive;
}

// Grava o estado final: linha "w h" e depois h linhas de w caracteres '0'/'1'.
// Mesmo formato gravado pela versao OpenCL (comparavel com cmp/diff).
static void dump_universe(void *u, int w, int h, const char *path)
{
   unsigned (*univ)[w] = u;
   FILE *f = fopen(path, "w");
   if (!f) {
      perror("Erro ao criar o arquivo de dump");
      return;
   }
   fprintf(f, "%d %d\n", w, h);
   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         fputc(univ[y][x] ? '1' : '0', f);
      }
      fputc('\n', f);
   }
   fclose(f);
}

// A função game recebe largura, altura da matriz e o total de iterações 
void game(int w, int h, int max_iter, const char *dump_path)
{
   unsigned (*univ)[w] = malloc(sizeof(unsigned) * (size_t)w * (size_t)h);
   unsigned (*nw)[w]   = malloc(sizeof(unsigned) * (size_t)w * (size_t)h);
   if (!univ || !nw) {
      fprintf(stderr, "Falha de alocacao (%dx%d)\n", w, h);
      exit(EXIT_FAILURE);
   }
   
   // Inicializa o tabuleiro: Uma cruz perfeita cruzando a matriz ao meio
   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         // Ativa a célula se ela estiver na linha do meio OU na coluna do meio
         if (x == w / 2 || y == h / 2) {
            univ[y][x] = 1;
         } else {
            univ[y][x] = 0;
         }
      }
   }
   
   // const int save_interval = 500; // Salva a cada 500 iterações
   for (int iter = 0; iter <= max_iter; iter++) {
      // if (iter % save_interval == 0) {
      //    save_pbm(univ, w, h, iter);
      // }
      evolve(univ, nw, w, h);
   }
   int final_alive = count_alive(univ, w, h);
   printf("Total de celulas vivas ao final: %d\n", final_alive);

   if (dump_path) dump_universe(univ, w, h, dump_path);

   free(univ);
   free(nw);
}

int main(int argc, char **argv)
{
   int w = 500;
   int h = 500;
   int max_iter = 50000;
   const char *dump_path = NULL;

   if (argc != 1 && argc != 4 && argc != 5) {
      fprintf(stderr, "Uso: %s [largura altura iteracoes [arquivo_dump]]\n", argv[0]);
      return 1;
   }
   if (argc >= 4) {
      w = atoi(argv[1]);
      h = atoi(argv[2]);
      max_iter = atoi(argv[3]);
      if (w < 1 || h < 1 || max_iter < 0) {
         fprintf(stderr, "Parametros invalidos\n");
         return 1;
      }
   }
   if (argc == 5) dump_path = argv[4];

   struct timespec start, end;

   // Captura o tempo EXATAMENTE ANTES do processamento iniciar
   clock_gettime(CLOCK_MONOTONIC, &start);

   game(w, h, max_iter, dump_path);

   // Captura o tempo EXATAMENTE APÓS o processamento terminar
   clock_gettime(CLOCK_MONOTONIC, &end);

   double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

   printf("Tempo interno de execucao: %f segundos\n", time_taken);
   
   return 0;
}
