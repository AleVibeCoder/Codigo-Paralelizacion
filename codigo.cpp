#include <stdio.h>
#include <time.h>
int main() {
clock_t inicio, fin;
double time_ms;
inicio = clock();
int suma=0;
for(int i=0; i<4000; i++){
suma = suma + i;
printf("%d \n", i);
}
printf("La suma es: %d \n", suma);
fin = clock();
time_ms = ((double)(fin -inicio)/CLOCKS_PER_SEC)*1000;
printf("Tiempo : %.2f milisegundos", time_ms);
return 0;
}
