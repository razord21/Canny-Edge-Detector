#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int read_pgm_image(char *infilename, unsigned char **image, int *rows, int *cols)
{
   FILE *fp; char buf[71];
   if(infilename == NULL) fp = stdin; else {
      if((fp = fopen(infilename, "r")) == NULL){ fprintf(stderr, "Error reading %s\n", infilename); return 0; }
   }
   fgets(buf, 70, fp);
   if(strncmp(buf,"P5",2) != 0){ fprintf(stderr, "The file %s is not PGM\n", infilename); if(fp != stdin) fclose(fp); return 0; }
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');
   sscanf(buf, "%d %d", cols, rows);
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');
   if(((*image) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){ fprintf(stderr, "Memory allocation failure\n"); if(fp != stdin) fclose(fp); return 0; }
   if((*rows) != fread((*image), (*cols), (*rows), fp)){ fprintf(stderr, "Error reading image data\n"); if(fp != stdin) fclose(fp); free((*image)); return 0; }
   if(fp != stdin) fclose(fp); return 1;
}

int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval)
{
   FILE *fp; if(outfilename == NULL) fp = stdout; else { if((fp = fopen(outfilename, "w")) == NULL){ fprintf(stderr, "Error writing %s\n", outfilename); return 0; } }
   fprintf(fp, "P5\n%d %d\n", cols, rows);
   if(comment != NULL) if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
   fprintf(fp, "%d\n", maxval);
   if(rows != fwrite(image, cols, rows, fp)){ fprintf(stderr, "Error writing image data\n"); if(fp != stdout) fclose(fp); return 0; }
   if(fp != stdout) fclose(fp); return 1;
}
