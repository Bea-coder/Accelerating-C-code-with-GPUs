#include <stdio.h>
#include <math.h>

#define NDIM 3
#define MAX 200

void write_output(char fname[MAX], int * in, unsigned long int size){

 int t;
 FILE *fp;

   fp = fopen(fname, "w");
   
   for (t=0; t<size; t++)
           fprintf(fp, "%d\n",in[t]);
      
   fclose(fp);
   
   printf("\nData in %s\n",fname);

}
