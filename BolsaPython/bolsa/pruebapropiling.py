import pandas as pd
from pandas_profiling import ProfileReport

### PANDAS PROFILING
df_completo = pd.read_csv(filepath_or_buffer='/bolsa/pasado/subgrupos/SG_19/COMPLETO.csv', sep='|')
prof = ProfileReport(df_completo.sample(n=10000))
prof.to_file(outputfile='/bolsa/pasado/subgrupos/SG_19/COMPLETO_profiling.html')


df_reducido = pd.read_csv(filepath_or_buffer='/bolsa/pasado/subgrupos/SG_19/REDUCIDO.csv', sep='|')
prof = ProfileReport(df_reducido)
prof.to_file(outputfile='/bolsa/pasado/subgrupos/SG_19/REDUCIDO_profiling.html')


