**Tabla T.** Sensibilidad a la inclusión de ANO (conjunto de prueba)

                 Especificación     R²    MAE   RMSE  Reducción MAE vs pers. (%)  p Holm (todos)  p Holm (con historial)  p Holm X1 en modelo final
   Principal (sin ANO, 16 var.) 0.7348 0.5007 1.0224                       5.096          0.1573                  0.1708                     1.0000
Sensibilidad (con ANO, 17 var.) 0.7384 0.4966 1.0155                       5.871          0.0104                  0.0160                     0.0098

**Tabla U.** Ablación en validación temporal con y sin ANO

                        ΔMAE sin ANO  p Holm sin ANO  ΔMAE con ANO  p Holm con ANO
Dimension                                                                         
X1_Temporal                  -0.0032          1.0000        0.0006          0.8457
X2_Productiva                -0.0017          1.0000       -0.0002          0.8457
X3_Espacial_Producto                 0.0540          0.0059        0.0567          0.0039
X4_Economica_Historica        0.0462          0.0039        0.0449          0.0039