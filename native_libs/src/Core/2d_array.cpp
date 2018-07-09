#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <vector>

#include "Common.h"

extern "C"
{

EXPORT char **join(char **fst_mat, char **snd_mat, size_t fst_rows, size_t fst_cols, size_t snd_rows, size_t snd_cols){
    size_t cols = ((fst_cols > snd_cols) ? fst_cols : snd_cols);
    char **new_mat = (char**)malloc((fst_rows+snd_rows)*cols*sizeof(*fst_mat));
    for(size_t r = 0; r < fst_rows+snd_rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
        {
            if (r < fst_rows) 
			{
                int elem = (r * cols + c);
                int new_index = r * cols + c ;
                char *tmp = fst_mat[elem];
                memcpy(&new_mat[new_index], &fst_mat[elem], sizeof(*fst_mat));
            } 
			else 
			{
                int elem = ((r - fst_rows) * cols + c);
                int new_index = r * cols + c ;
                char *tmp = snd_mat[elem];
                memcpy(&new_mat[new_index], &snd_mat[elem], sizeof(*snd_mat));
            }


        }
    }
    return new_mat;
}


} // extern "C"