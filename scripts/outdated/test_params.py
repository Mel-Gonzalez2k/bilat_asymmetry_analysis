import pandas as pd
df = pd.read_csv(r"E:\bilat_asymmetry_analysis\data\TeLC_Silencing\WA023\20260413\Post_Day7\right\right_whisker1_angle_filt.csv")
print(df["Data"].describe())
print(df["Data"].quantile([.01,.5,.99]))