#include <stdio.h>
#include <stdlib.h>

#define VERBOSE 1
#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

static int times = 0;

static void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval, int cols)
{
   times++;
   short *tempmagptr;
   unsigned char *tempmapptr;
   int i;
   int x[8] = {1,1,0,-1,-1,-1,0,1};
   int y[8] = {0,1,1,1,0,-1,-1,-1};
   for(i=0;i<8;i++){
      tempmapptr = edgemapptr - y[i]*cols + x[i];
      tempmagptr = edgemagptr - y[i]*cols + x[i];
      if((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval)){
         *tempmapptr = (unsigned char) EDGE;
         follow_edges(tempmapptr,tempmagptr, lowval, cols);
      }
   }
}

void apply_hysteresis(short int *mag, unsigned char *nms, int rows, int cols, float tlow, float thigh, unsigned char *edge)
{
   int r, c, pos, numedges, lowcount, highcount, lowthreshold, highthreshold, i, hist[32768];
   short int maximum_mag;
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
         if(nms[pos] == POSSIBLE_EDGE) edge[pos] = POSSIBLE_EDGE; else edge[pos] = NOEDGE;
      }
   }
   for(r=0,pos=0;r<rows;r++,pos+=cols){ edge[pos] = NOEDGE; edge[pos+cols-1] = NOEDGE; }
   pos = (rows-1) * cols;
   for(c=0;c<cols;c++,pos++){ edge[c] = NOEDGE; edge[pos] = NOEDGE; }

   for(r=0;r<32768;r++) hist[r] = 0;
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++) if(edge[pos] == POSSIBLE_EDGE) hist[mag[pos]]++;
   }

   for(r=1,numedges=0;r<32768;r++){ if(hist[r] != 0) maximum_mag = r; numedges += hist[r]; }
   highcount = (int)(numedges * thigh + 0.5);
   r = 1; numedges = hist[1];
   while((r<(maximum_mag-1)) && (numedges < highcount)){ r++; numedges += hist[r]; }
   highthreshold = r;
   lowthreshold = (int)(highthreshold * tlow + 0.5);
   if(VERBOSE){
      printf("The input low and high fractions of %f and %f computed to\n", tlow, thigh);
      printf("magnitude of the gradient threshold values of: %d %d\n", lowthreshold, highthreshold);
   }
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++){
         if((edge[pos] == POSSIBLE_EDGE) && (mag[pos] >= highthreshold)){
            edge[pos] = EDGE;
            follow_edges((edge+pos), (mag+pos), lowthreshold, cols);
         }
      }
   }
   printf("%d\n",times);
   for(r=0,pos=0;r<rows;r++){
      for(c=0;c<cols;c++,pos++) if(edge[pos] != EDGE) edge[pos] = NOEDGE;
   }
}

void non_max_supp(short *mag, short *gradx, short *grady, int nrows, int ncols, unsigned char *result)
{
    int rowcount, colcount,count;
    short *magrowptr,*magptr;
    short *gxrowptr,*gxptr;
    short *gyrowptr,*gyptr,z1,z2;
    short m00,gx,gy;
    float mag1,mag2,xperp,yperp;
    unsigned char *resultrowptr, *resultptr;

    for(count=0,resultrowptr=result,resultptr=result+ncols*(nrows-1); count<ncols; resultptr++,resultrowptr++,count++){
        *resultrowptr = *resultptr = (unsigned char) 0;
    }
    for(count=0,resultptr=result,resultrowptr=result+ncols-1; count<nrows; count++,resultptr+=ncols,resultrowptr+=ncols){
        *resultptr = *resultrowptr = (unsigned char) 0;
    }

    for(rowcount=1,magrowptr=mag+ncols+1,gxrowptr=gradx+ncols+1, gyrowptr=grady+ncols+1,resultrowptr=result+ncols+1; rowcount<nrows-2; rowcount++,magrowptr+=ncols,gyrowptr+=ncols,gxrowptr+=ncols, resultrowptr+=ncols){
      for(colcount=1,magptr=magrowptr,gxptr=gxrowptr,gyptr=gyrowptr, resultptr=resultrowptr;colcount<ncols-2; colcount++,magptr++,gxptr++,gyptr++,resultptr++){
         m00 = *magptr;
         if(m00 == 0){ *resultptr = (unsigned char) NOEDGE; }
         else{ xperp = -(gx = *gxptr)/((float)m00); yperp = (gy = *gyptr)/((float)m00); }

         if(gx >= 0){
            if(gy >= 0){
               if (gx >= gy){
                  z1 = *(magptr - 1); z2 = *(magptr - ncols - 1);
                  mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                  z1 = *(magptr + 1); z2 = *(magptr + ncols + 1);
                  mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
               } else {
                  z1 = *(magptr - ncols); z2 = *(magptr - ncols - 1);
                  mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                  z1 = *(magptr + ncols); z2 = *(magptr + ncols + 1);
                  mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
               }
            } else {
               if (gx >= -gy){
                  z1 = *(magptr - 1); z2 = *(magptr + ncols - 1);
                  mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                  z1 = *(magptr + 1); z2 = *(magptr - ncols + 1);
                  mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
               } else {
                  z1 = *(magptr + ncols); z2 = *(magptr + ncols - 1);
                  mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;
                  z1 = *(magptr - ncols); z2 = *(magptr - ncols + 1);
                  mag2 = (z1 - z2)*xperp  + (m00 - z1)*yperp;
               }
            }
         } else {
            if ((gy = *gyptr) >= 0){
               if (-gx >= gy){
                  z1 = *(magptr + 1); z2 = *(magptr - ncols + 1);
                  mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                  z1 = *(magptr - 1); z2 = *(magptr + ncols - 1);
                  mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
               } else {
                  z1 = *(magptr - ncols); z2 = *(magptr - ncols + 1);
                  mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                  z1 = *(magptr + ncols); z2 = *(magptr + ncols - 1);
                  mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
               }
            } else {
               if (-gx > -gy){
                  z1 = *(magptr + 1); z2 = *(magptr + ncols + 1);
                  mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                  z1 = *(magptr - 1); z2 = *(magptr - ncols - 1);
                  mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
               } else {
                  z1 = *(magptr + ncols); z2 = *(magptr + ncols + 1);
                  mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                  z1 = *(magptr - ncols); z2 = *(magptr - ncols - 1);
                  mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
               }
            }
         }

         if ((mag1 > 0.0) || (mag2 > 0.0)) *resultptr = (unsigned char) NOEDGE;
         else { if (mag2 == 0.0) *resultptr = (unsigned char) NOEDGE; else *resultptr = (unsigned char) POSSIBLE_EDGE; }
      }
    }
}
