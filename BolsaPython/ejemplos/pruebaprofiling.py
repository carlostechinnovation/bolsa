import pandas as pd
from pandas_profiling import ProfileReport

# print("COMPLETO - Profiling...")
# df_completo = pd.read_csv(filepath_or_buffer="/bolsa/pasado/subgrupos/SG_16/COMPLETO.csv", sep="|")
# if len(df_completo) > 2000:
#     prof = ProfileReport(df_completo.sample(n=2000))
# else:
#     prof = ProfileReport(df_completo)
#
# prof.to_file(output_file="/bolsa/pasado/subgrupos/SG_16/COMPLETO_profiling.html")

print("REDUCIDO - Profiling...")
df_reducido = pd.read_csv(filepath_or_buffer="/bolsa/pasado/subgrupos/SG_16/REDUCIDO.csv", sep="|")
if len(df_reducido) > 2000:
    prof = ProfileReport(df_reducido.sample(n=2000))
else:
    prof = ProfileReport(df_reducido)

prof.to_file(output_file="/bolsa/pasado/subgrupos/SG_16/REDUCIDO_profiling.html")


