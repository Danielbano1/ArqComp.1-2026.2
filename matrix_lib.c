// Cabeçalhos das bibliotecas padrão da linguagem C
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

// Cabeçalhos das minhas bibliotecas
#include "matrix_lib.h"

void matrix_zero_avx(float *values, unsigned long int rows, unsigned long int cols) {
    unsigned long int n = rows * cols;
    unsigned long int i = 0;

    __m256 zero = _mm256_setzero_ps();

    for (; i + 8 <= n; i += 8) {
        _mm256_store_ps(&values[i], zero);
    }

    // resto (se n não for múltiplo de 8)
    for (; i < n; i++) {
        values[i] = 0.0f;
    }
}

/*
    Ainda falta tratar os possiveis erros.
*/
int scalar_matrix_mult(float scalar_value, matrix *m, matrix *r)
{
    matrix_zero_avx(r->values, r->rows, r->cols);

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
        _mm256_store_ps(&r->values[i], v); 
    }

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
    // Verifica se a multiplicação é possível: colunas de M1 deve ser igual a linhas de M2
    if (m1->cols != m2->rows) {
        return 1;
    }
    
    int qtd_colunas_m1 = m1->cols;
    int qtd_colunas_m2 = m2->cols;
    int corte_linha_m2 = qtd_colunas_m2 / 8;
    unsigned long int m1_row_offset;
    unsigned long int r_row_offset;
    int qtd_linhas = m1->rows;
    float *m1_values = m1->values;
    float *m2_values = m2->values;
    float *r_values = r->values;

    matrix_zero_avx(r_values, r->rows, r->cols); // zera antes de acumular com FMA
    
    // itera pela quantidade de linhas de m1
    for (int i = 0; i < qtd_linhas; i++)
    {
        m1_row_offset = i * qtd_colunas_m1;  // offset em M1
        r_row_offset  = i * qtd_colunas_m2;  // offset em R (mesmas colunas que M2)

        // itera pela quantidade de colunas de m1 (linhas de m2)
        for (int j = 0; j < qtd_colunas_m1; j++)
        {
            // Pega m1[i][j] e replica em todos os 8 lanes
            __m256 m1_j = _mm256_set1_ps(m1_values[m1_row_offset + j]);

            // divide as linhas de m2 em 8 colunas para poder pegar o vetor
            for (int k = 0; k < corte_linha_m2; k++)
            {
                __m256 m2_vec = _mm256_load_ps(&m2_values[j * qtd_colunas_m2 + k * 8]);
                __m256 r_vec = _mm256_load_ps(&r_values[r_row_offset + k * 8]);

                r_vec = _mm256_fmadd_ps(m1_j, m2_vec, r_vec);

                _mm256_store_ps(&r_values[r_row_offset + k * 8], r_vec);
            }
        }
    }
    
    return 0;
}


