#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

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

void write_csv(const char *filename, char **mat, size_t rows, size_t cols){
    FILE *fp = fopen(filename, "w+");
    if(!fp){
        fprintf(stderr, "%s can't open in %s\n", filename, __func__);
        perror("fopen");
        abort();

    }
    for(size_t r = 0; r < rows; ++r){
        for(size_t c = 0; c < cols; ++c){
            int record = r*cols+c;
            fprintf(fp, "\"" );
            fprintf(fp, "%s", (mat[record] ? mat[record] : ""));
            fprintf(fp, "\"" );
            if (c !=cols-1) {
                fprintf(fp, ",");
            }

        }
        fprintf(fp,"%s","\n");
    }
    fclose(fp);

}

int main(void){
    size_t rows, cols;
    int error;
    char **mat = read_csv("./data/simple_empty.csv", &rows, &cols, &error);
    int j[]  = {2,1};
    write_csv("./data/simple2.csv", mat, rows, cols);

    return 0;
}
