#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

char **copy_columns(char **mat, size_t rows, size_t cols, size_t stride_r, size_t stride_c, int number_of_cols, int *columns_to_copy)
{
    if (number_of_cols<1)
        return NULL;

    if (columns_to_copy == NULL)
        return NULL;

    if (mat == NULL)
        return NULL;

    char **new_mat = malloc(rows*number_of_cols*sizeof(*mat));

    for(size_t r = 0; r < rows; r++)
    {
        for(size_t c = 0; c < number_of_cols; c++)
        {

            // printf("%s: %d\n","r",r );
            // printf("%s: %d\n","c",c );
            int column =  columns_to_copy[c];
            int elem = (r * column + column * stride_c);
            // printf("%s: %d\n","elem",elem );
            int new_index = r * number_of_cols + c;
            // printf("%s: %d\n","new_index",new_index);
            char *tmp = mat[elem];
            // printf("%s\n", tmp ? tmp : "NULL");
            memcpy(&new_mat[new_index], &mat[elem], sizeof(*mat));

        }
    }
    return new_mat;
}

char **copy_rows(char **mat, size_t rows, size_t cols, size_t stride_r, size_t stride_c, int number_of_rows, int *rows_to_copy)
{
    if (number_of_rows<1)
        return NULL;

    if (rows_to_copy == NULL)
        return NULL;

    if (mat == NULL)
        return NULL;

    char **new_mat = malloc(number_of_rows*cols*sizeof(*mat));

    for(size_t r = 0; r < number_of_rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
        {
            int row =  rows_to_copy[r];
            int elem = (row * stride_r + c * stride_c);
            // printf("%s: %d\n","elem",elem );
            int new_index = r * stride_r + c * stride_c;
            // printf("%s: %d\n","new_index",new_index);
            char *tmp = mat[elem];
            // printf("%s\n", tmp ? tmp : "NULL");
            memcpy(&new_mat[new_index], &mat[elem], sizeof(*mat));

        }
    }
    return new_mat;
}

char **drop_row(char **mat, size_t rows, size_t cols, size_t stride_r, size_t stride_c, int row_to_drop)
{
    if (row_to_drop>rows)
        return NULL;

    if (mat == NULL)
        return NULL;

    int rows_to_copy[rows-1];
        for(int i = 0; i < row_to_drop; i++) {
                rows_to_copy[i] = i;
        }
        for(int i = row_to_drop+1; i <= rows; i++) {
                rows_to_copy[i-1] = i;
        }

    char **new_mat = malloc((rows-1)*cols*sizeof(*mat));

    for(size_t r = 0; r < rows-1; r++)
    {
        for(size_t c = 0; c < cols; c++)
        {
            int row =  rows_to_copy[r];
            int elem = (row * stride_r + c * stride_c);
            // printf("%s: %d\n","elem",elem );
            int new_index = r * stride_r + c * stride_c;
            // printf("%s: %d\n","new_index",new_index);
            char *tmp = mat[elem];
            // printf("%s\n", tmp ? tmp : "NULL");
            memcpy(&new_mat[new_index], &mat[elem], sizeof(*mat));

        }
    }
    return new_mat;
}

char **transpose(char **mat, size_t rows, size_t cols, size_t stride_r, size_t stride_c)
{
    if (mat == NULL)
        return NULL;

    char **new_mat = malloc(rows*cols*sizeof(*mat));

    for(size_t r = 0; r < rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
        {
            int elem = (r * stride_r + c * stride_c);
            // printf("%s: %d\n","elem",elem );
            int new_index =  c * rows + r ;
            // printf("%s: %d\n","new_index",new_index);
            char *tmp = mat[elem];
            // printf("%s\n", tmp ? tmp : "NULL");
            memcpy(&new_mat[new_index], &mat[elem], sizeof(*mat));

        }
    }
    return new_mat;
}
