**Parameters assigned**

/# Assign the experiment design table to 'file_in'
file_in = 'design_table_3.xlsx'
/# Assign the fluorescence filename to 'file_out'
file_out = 'experiment_3.xlsx'

/# dimensions of the plate
/# given below for a 384 well plate (16x24)
total_rows = 16
total_columns = 24

/# fluorescence reading labels as mentioned in the output file
labels = ["sfGFP20", "MGA80", "sfGFP40"]

/# Active sheet in the fluorescence file
sheet_name = "Result sheet"

/# features to plot against each other 
expt = ["DNA N30", "DNA N35"]
/# expt = ["DNA N32", "DNA N35"]

/# Are the other reagents are also varying?(comment the line which does not apply)
answer = "no"
