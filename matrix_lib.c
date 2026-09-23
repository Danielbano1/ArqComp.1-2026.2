// Cabeçalhos das bibliotecas padrão da linguagem C
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

// Cabeçalhos das minhas bibliotecas
#include "matrix_lib.h"

// Novos tipos
typedef struct
{
    matrix *m1;
    matrix *m2;
    matrix *r;
    int thread_id;
} pthread_args;

// Variáveis globais
#define N_THREADS 16

void *matrix_worker(void *args)
{
    pthread_args *thread_args = (pthread_args *)args;
    int qtd_colunas_m1 = thread_args->m1->cols;
    int qtd_colunas_m2 = thread_args->m2->cols;
    int corte_linha_m2 = qtd_colunas_m2 / 8;
    int linha_inicial;
    int qtd_linhas;
    int thread_id;
    int sobraram; /// qtd de linhas que sobraram porque a matriz tem n_linhas multiplo de n_threads
    int linha;    /// número da linha que sobrou que eu vou fazer
    float *m1_values = thread_args->m1->values;
    float *m2_values = thread_args->m2->values;
    float *r_values = thread_args->r->values;

    thread_id = thread_args->thread_id;
    qtd_linhas = thread_args->r->rows / N_THREADS;
    sobraram = thread_args->r->rows % N_THREADS;

    // itera pela quantidade de linhas que cada thread vai fazer
    for (int i = 0; i < qtd_linhas; i++)
    {
        // Linha que a thread vai fazer
        linha_inicial = thread_id + (i * qtd_colunas_m1);
        // itera pela quantidade de colunas de m1 (linhas de m2)
        for (int j = 0; j < qtd_colunas_m1; j++)
        {
            // Pega m1[i][j] e replica em todos os 8 lanes
            __m256 m1_j = _mm256_set1_ps(m1_values[linha_inicial + j]);
            // divide as linhas de m2 em 8 colunas para poder pegar o vetor
            for (int k = 0; k < corte_linha_m2; k + 8)
            {
                __m256 m2_vec = _mm256_load_ps(&m2_values[(j * qtd_colunas_m2) + k]);
                __m256 r_vec = _mm256_load_ps(&r_values[linha_inicial + k]);

                r_vec = _mm256_fmadd_ps(m1_j, m2_vec, r_vec);

                _mm256_store_ps(&r_values[linha_inicial + k], r_vec);
            }
        }
    }

    // implementar a multiplicação de uma das linhas que sobraram
    if (thread_id < sobraram)
    {
        // linha que sobrou que eu vou fazer
        linha = qtd_linhas * N_THREADS + thread_id;

        // itera pela quantidade de colunas de m1 (linhas de m2)
        for (int j = 0; j < qtd_colunas_m1; j++)
        {
            // Pega m1[linha][j] e replica em todos os 8 lanes
            __m256 m1_j = _mm256_set1_ps(m1_values[linha + j]);
            // divide as linhas de m2 em 8 colunas para poder pegar o vetor
            for (int k = 0; k < corte_linha_m2; k + 8)
            {
                __m256 m2_vec = _mm256_load_ps(&m2_values[(j * qtd_colunas_m2) + k]);
                __m256 r_vec = _mm256_load_ps(&r_values[linha + k]);

                r_vec = _mm256_fmadd_ps(m1_j, m2_vec, r_vec);

                _mm256_store_ps(&r_values[linha + k], r_vec);
            }
        }
    }
}

/*
    Ainda falta tratar os possiveis erros. E apagar a versao sem alinhamento de memoria
*/
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r)
{
    // sem alinhamento de memoria
    /////////////////////////////////////////////
    unsigned long int n = m->rows * m->cols;
    unsigned long int i = 0;

    __m256 vscalar = _mm256_set1_ps(scalar_value);

    // Processa 8 floats por vez (AVX)
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(&m->values[i]);
        v = _mm256_mul_ps(v, vscalar);
        _mm256_storeu_ps(&r->values[i], v); // conferir se a matriz r esta apta para receber os valores
    }
    // Trata os elementos restantes (resto da divisão por 8)
    for (; i < n; i++)
    {
        r->values[i] = m->values[i] * scalar_value;
    }
    /////////////////////////////////////////////////
    // com alinhamento de memoria
    /////////////////////////////////////////////
    unsigned long int n = m->rows * m->cols;
    unsigned long int i = 0;

    __m256 vscalar = _mm256_set1_ps(scalar_value);

    // Processa 8 floats por vez (AVX)
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_load_ps(&m->values[i]);
        v = _mm256_mul_ps(v, vscalar);
        _mm256_store_ps(&r->values[i], v); // conferir se a matriz r esta apta para receber os valores
    }
    // Trata os elementos restantes (resto da divisão por 8)
    for (; i < n; i++)
    {
        r->values[i] = m->values[i] * scalar_value;
    }
    /////////////////////////////////////////////////
    return 0;
}

/**
 * @param m1 matrix M1
 * @param m2 matrix M2
 * @param r matrix R
 * @returns error code
 *
 * Calcula R = M1 x M2
 *
 * Códigos de erro:
 *
 * - 1 - Erro ao criar thread
 *
 * - 2 - Erro ao esperar por thread
 *
 */
int matrix_matrix_mult(matrix *m1, matrix *m2, matrix *r)
{
    pthread_attr_t attr;
    pthread_args thread_args[N_THREADS];
    pthread_t thread_id[N_THREADS];
    void *status;

    int rc; /// valor de retorno das funções pthread

    // atributo das threads
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    // Criar threads
    for (int i = 0; i < N_THREADS; i++)
    {
        thread_args[i].m1 = m1;
        thread_args[i].m2 = m2;
        thread_args[i].r = r;
        thread_args[i].thread_id = i;
        rc = pthread_create(&thread_id[i], &attr, matrix_worker, (void *)(&thread_args[i]));
        if (rc)
        {
            fprintf(stderr, "[ERROR %ld] Erro #%d ao criar threads\n", __LINE__, rc);
            exit(1);
        }
    }

    // destruir o atributo das threads
    pthread_attr_destroy(&attr);

    // Esperar que todas as threads criadas terminem
    for (int i = 0; i < N_THREADS; i++)
    {
        rc = pthread_join(thread_id[i], &status);
        if (rc)
        {
            fprintf(stderr, "[ERROR %ld] return code from pthread_join() is %d\n", __LINE__, rc);
            exit(2);
        }
    }

    return 0;
}
