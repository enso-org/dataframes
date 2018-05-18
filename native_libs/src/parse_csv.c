#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

double luna_scanf(char *data)
{
    double d;
    sscanf(data,"%lf", &d);
    return d;
}


//https://tools.ietf.org/html/rfc4180
char *getCSVField(FILE *fp, char separator, int *state){
    int ch = fgetc(fp);

    if(ch == EOF)
        return NULL;

    size_t size = 1, index = 0;
    char *field = malloc(size);
    bool quoted_in = false;

    for(;ch != EOF; ch = fgetc(fp)){
        if(ch == '"'){
            if(quoted_in){
                int prefetch = fgetc(fp);
                if(prefetch == '"'){
                    ch = prefetch;
                } else {
                    quoted_in = false;
                    ungetc(prefetch, fp);
                    continue;
                }
            } else {
                quoted_in = true;
                continue;
            }
        } else if(!quoted_in && (ch == separator || ch == '\n')){
            break;
        }
        field[index++] = ch;
        char *temp = realloc(field, ++size);
        if(!temp){
            perror("realloc:");
            free(field);
            exit(EXIT_FAILURE);
        }
        field = temp;
    }
    field[index] = 0;
    *state = ch;
    if(quoted_in){
        fprintf(stderr, "The quotes is not closed.\n");
        free(field);
        return NULL;
    }
    return field;
}

char **read_csv(const char *filename, size_t *rows, size_t *cols, int* error){
    *rows = *cols = 0;

    FILE *fp = fopen(filename, "r");
    if(!fp){
        fprintf(stderr, "%s can't open in %s\n", filename, __func__);
        perror("fopen");
        return NULL;
    }

    char *field;
    int state;
    size_t r = 0, c = 0;
    char **mat = NULL;


    while((field = getCSVField(fp, ',', &state))){
        if(c == 0){
            if(r>0){
                // printf ("%ld\n", (((r+1) * *cols)*sizeof(*mat)));
                char** newmat = realloc(mat, ((r+1) * *cols)*sizeof(*mat));
                if (newmat == NULL)
                    goto err;
                else
                    mat = newmat;
            }
            else{
                // printf ("%ld\n", sizeof(*mat));
                char **newmat = realloc(mat, sizeof(*mat));
                if (newmat == NULL)
                    goto err;
                else
                    mat = newmat;
            }
        }
        if (r == 0){
            // printf("realloc: %ld\n", (c+1)*sizeof(*mat));
            char **newmat = realloc(mat, (c+1)*sizeof(*mat));
            if (newmat == NULL)
                goto err;
            else
                mat = newmat;
        }
        int index = r * (*cols) + c;
        // printf("index: %ld\n", index);
        if (strncmp(field,"",1)==0)
        {
            mat[index] = NULL;
        }
        else
        {
            mat[index] = field;
        }
        c++;
        if(state == '\n' || state == EOF){
            if(*cols == 0){
                *cols = c;
            } else if(c != *cols){
                *error = 1;
                return NULL;
            }
            c  = 0;
            *rows = ++r;
        }
    }
    fclose(fp);

    return mat;

err:
    free(mat);
    *error = 2;
    return NULL;
}

void mat_delete(void **mat)
{
    free(mat);
}

char **copy_column(char **mat, size_t rows, size_t cols, size_t stride_r, size_t stride_c, int offset)
{
    char **new_mat = malloc(rows*cols*sizeof(*mat));
    for(size_t r = 0; r < rows; r++)
    {
        for(size_t c = 0; c < cols; c++)
        {
            int elem = (r * stride_r + c * stride_c + offset)  ;
            int new_index = r;
            memcpy(&new_mat[new_index], &mat[elem], sizeof(*mat));

        }
    }
    return new_mat;
}

#include <ctype.h>

char *trim(char *s){
    if(!s || !*s)
        return s;

    char *from, *to;

    for(from = s; *from && isspace((unsigned char)*from); ++from);
    for(to = s; *from;){
        *to++ = *from++;
    }
    *to = 0;
    while(s != to && isspace((unsigned char)to[-1])){
        *--to = 0;
    }
    return s;
}

int main(void){
    size_t rows, cols;
    int error;
    char **mat = read_csv("/home/sylwia/project/csv-parser/test/data/simple_empty.csv", &rows, &cols, &error);
    printf("%ld,\n",cols);
    printf("%ld,\n",rows);
    char **column = copy_column(**mat,rows,cols,cols,1,1);

    for(size_t r = 0; r < rows; ++r){
        for(size_t c = 0; c < 1; ++c){
            if(c)
                putchar(',');
            printf("%s", (column[r*cols+c]));
        }
        puts("");
    }
    free(mat);
    return 0;
}
