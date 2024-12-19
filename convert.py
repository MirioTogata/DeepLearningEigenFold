import numpy as np

input_data = """
./data/pdb_chains/8R/8R2Y.A.pdb ELBO_Y 0.3490101881297962 {'rmsd': 0.805, 'tm': 0.9855, 'gdt_ts': 0.9731, 'gdt_ha': 0.874, 'lddt': 0.9549}
./data/pdb_chains/8R/8R2Y.A.pdb ELBO_Y 0.3442782111839139 {'rmsd': 0.77, 'tm': 0.9869, 'gdt_ts': 0.9779, 'gdt_ha': 0.9, 'lddt': 0.9598}
./data/pdb_chains/8R/8R2Y.A.pdb ELBO_Y 0.3518782557545796 {'rmsd': 0.792, 'tm': 0.9854, 'gdt_ts': 0.9798, 'gdt_ha': 0.8644, 'lddt': 0.9495}
./data/pdb_chains/8R/8R2Y.A.pdb ELBO_Y 0.3131292197639978 {'rmsd': 0.76, 'tm': 0.9862, 'gdt_ts': 0.976, 'gdt_ha': 0.8731, 'lddt': 0.9528}
./data/pdb_chains/8R/8R2Y.A.pdb ELBO_Y 0.3625339248980817 {'rmsd': 0.785, 'tm': 0.9857, 'gdt_ts': 0.9817, 'gdt_ha': 0.8654, 'lddt': 0.9482}
./data/pdb_chains/8R/8RSF.A.pdb ELBO_Y 0.06213836959007303 {'rmsd': 3.207, 'tm': 0.9728, 'gdt_ts': 0.9668, 'gdt_ha': 0.871, 'lddt': 0.9292}
./data/pdb_chains/8R/8RSF.A.pdb ELBO_Y 0.056216857478180345 {'rmsd': 3.401, 'tm': 0.9717, 'gdt_ts': 0.9677, 'gdt_ha': 0.8781, 'lddt': 0.9256}
./data/pdb_chains/8R/8RSF.A.pdb ELBO_Y 0.10150311992989053 {'rmsd': 3.275, 'tm': 0.9718, 'gdt_ts': 0.9668, 'gdt_ha': 0.8719, 'lddt': 0.9247}
./data/pdb_chains/8R/8RSF.A.pdb ELBO_Y 0.07961289362398927 {'rmsd': 3.595, 'tm': 0.9671, 'gdt_ts': 0.9606, 'gdt_ha': 0.8396, 'lddt': 0.9117}
./data/pdb_chains/8R/8RSF.A.pdb ELBO_Y 0.0618696719821872 {'rmsd': 3.236, 'tm': 0.9732, 'gdt_ts': 0.9722, 'gdt_ha': 0.8799, 'lddt': 0.9301}
./data/pdb_chains/8T/8T68.A.pdb ELBO_Y -0.3540616337438208 {'rmsd': 4.748, 'tm': 0.7339, 'gdt_ts': 0.5927, 'gdt_ha': 0.3879, 'lddt': 0.6832}
./data/pdb_chains/8T/8T68.A.pdb ELBO_Y -0.31205696527030585 {'rmsd': 4.463, 'tm': 0.7442, 'gdt_ts': 0.6013, 'gdt_ha': 0.4019, 'lddt': 0.696}
./data/pdb_chains/8T/8T68.A.pdb ELBO_Y -0.43019116625917514 {'rmsd': 5.203, 'tm': 0.746, 'gdt_ts': 0.6218, 'gdt_ha': 0.4332, 'lddt': 0.6664}
./data/pdb_chains/8T/8T68.A.pdb ELBO_Y -0.31451160055500743 {'rmsd': 4.045, 'tm': 0.7728, 'gdt_ts': 0.6185, 'gdt_ha': 0.417, 'lddt': 0.6872}
./data/pdb_chains/8T/8T68.A.pdb ELBO_Y -0.3091037312653809 {'rmsd': 4.62, 'tm': 0.7396, 'gdt_ts': 0.5884, 'gdt_ha': 0.3825, 'lddt': 0.6736}
./data/pdb_chains/8U/8UWM.A.pdb ELBO_Y 0.2604438431351131 {'rmsd': 25.04, 'tm': 0.5403, 'gdt_ts': 0.5054, 'gdt_ha': 0.435, 'lddt': 0.8769}
./data/pdb_chains/8U/8UWM.A.pdb ELBO_Y 0.2905143356938018 {'rmsd': 25.021, 'tm': 0.5306, 'gdt_ts': 0.5, 'gdt_ha': 0.4224, 'lddt': 0.8746}
./data/pdb_chains/8U/8UWM.A.pdb ELBO_Y 0.26302104720598796 {'rmsd': 25.09, 'tm': 0.5382, 'gdt_ts': 0.5045, 'gdt_ha': 0.435, 'lddt': 0.88}
./data/pdb_chains/8U/8UWM.A.pdb ELBO_Y 0.24412348452887958 {'rmsd': 25.03, 'tm': 0.5379, 'gdt_ts': 0.5027, 'gdt_ha': 0.4341, 'lddt': 0.8869}
./data/pdb_chains/8U/8UWM.A.pdb ELBO_Y 0.2641067957861357 {'rmsd': 24.964, 'tm': 0.5374, 'gdt_ts': 0.5018, 'gdt_ha': 0.4323, 'lddt': 0.8866}
./data/pdb_chains/8V/8VZG.A.pdb ELBO_Y 0.3354230111664518 {'rmsd': 1.332, 'tm': 0.9699, 'gdt_ts': 0.9484, 'gdt_ha': 0.8123, 'lddt': 0.925}
./data/pdb_chains/8V/8VZG.A.pdb ELBO_Y 0.2824593169418147 {'rmsd': 1.203, 'tm': 0.9712, 'gdt_ts': 0.9436, 'gdt_ha': 0.8093, 'lddt': 0.9231}
./data/pdb_chains/8V/8VZG.A.pdb ELBO_Y 0.27040055835430865 {'rmsd': 1.193, 'tm': 0.9713, 'gdt_ts': 0.9426, 'gdt_ha': 0.8161, 'lddt': 0.9239}
./data/pdb_chains/8V/8VZG.A.pdb ELBO_Y 0.31240808763000993 {'rmsd': 1.268, 'tm': 0.9708, 'gdt_ts': 0.9484, 'gdt_ha': 0.8298, 'lddt': 0.9236}
./data/pdb_chains/8V/8VZG.A.pdb ELBO_Y 0.3025625752035978 {'rmsd': 1.16, 'tm': 0.972, 'gdt_ts': 0.9523, 'gdt_ha': 0.8161, 'lddt': 0.924}
./data/pdb_chains/8X/8X6E.A.pdb ELBO_Y -0.45610021280449586 {'rmsd': 15.418, 'tm': 0.3518, 'gdt_ts': 0.2301, 'gdt_ha': 0.1453, 'lddt': 0.3936}
./data/pdb_chains/8X/8X6E.A.pdb ELBO_Y -0.651256295733714 {'rmsd': 15.756, 'tm': 0.3421, 'gdt_ts': 0.2379, 'gdt_ha': 0.1652, 'lddt': 0.3999}
./data/pdb_chains/8X/8X6E.A.pdb ELBO_Y -0.5415246321938667 {'rmsd': 18.947, 'tm': 0.353, 'gdt_ts': 0.2431, 'gdt_ha': 0.1644, 'lddt': 0.3832}
./data/pdb_chains/8X/8X6E.A.pdb ELBO_Y -0.6116386523267107 {'rmsd': 14.511, 'tm': 0.3829, 'gdt_ts': 0.2543, 'gdt_ha': 0.1721, 'lddt': 0.4273}
./data/pdb_chains/8X/8X6E.A.pdb ELBO_Y -0.6803411339582257 {'rmsd': 20.058, 'tm': 0.3258, 'gdt_ts': 0.2024, 'gdt_ha': 0.1324, 'lddt': 0.3723}
./data/pdb_chains/8X/8XB1.A.pdb ELBO_Y -0.04601966658745046 {'rmsd': 1.238, 'tm': 0.9682, 'gdt_ts': 0.9079, 'gdt_ha': 0.7336, 'lddt': 0.8938}
./data/pdb_chains/8X/8XB1.A.pdb ELBO_Y -0.023845768351591266 {'rmsd': 1.67, 'tm': 0.9579, 'gdt_ts': 0.8931, 'gdt_ha': 0.7188, 'lddt': 0.8839}
./data/pdb_chains/8X/8XB1.A.pdb ELBO_Y -0.1484189254545553 {'rmsd': 4.647, 'tm': 0.9156, 'gdt_ts': 0.8717, 'gdt_ha': 0.7294, 'lddt': 0.8361}
./data/pdb_chains/8X/8XB1.A.pdb ELBO_Y 0.05732349961042902 {'rmsd': 1.222, 'tm': 0.9693, 'gdt_ts': 0.9095, 'gdt_ha': 0.7393, 'lddt': 0.8881}
./data/pdb_chains/8X/8XB1.A.pdb ELBO_Y -0.044226040130125176 {'rmsd': 1.578, 'tm': 0.9629, 'gdt_ts': 0.8972, 'gdt_ha': 0.736, 'lddt': 0.8866}
./data/pdb_chains/8X/8XWG.A.pdb ELBO_Y -0.09318592324675787 {'rmsd': 5.299, 'tm': 0.6999, 'gdt_ts': 0.5396, 'gdt_ha': 0.3441, 'lddt': 0.8227}
./data/pdb_chains/8X/8XWG.A.pdb ELBO_Y -0.014716608901926925 {'rmsd': 7.657, 'tm': 0.6083, 'gdt_ts': 0.4739, 'gdt_ha': 0.3141, 'lddt': 0.8164}
./data/pdb_chains/8X/8XWG.A.pdb ELBO_Y -0.052851016997769666 {'rmsd': 10.587, 'tm': 0.5931, 'gdt_ts': 0.4873, 'gdt_ha': 0.3386, 'lddt': 0.7858}
./data/pdb_chains/8X/8XWG.A.pdb ELBO_Y 0.006907882478929446 {'rmsd': 5.716, 'tm': 0.6939, 'gdt_ts': 0.534, 'gdt_ha': 0.36, 'lddt': 0.8077}
./data/pdb_chains/8X/8XWG.A.pdb ELBO_Y -0.09756247001293535 {'rmsd': 5.399, 'tm': 0.7511, 'gdt_ts': 0.5665, 'gdt_ha': 0.3616, 'lddt': 0.8028}
./data/pdb_chains/9B/9BA3.A.pdb ELBO_Y -0.03561789806746825 {'rmsd': 3.056, 'tm': 0.8835, 'gdt_ts': 0.8344, 'gdt_ha': 0.6609, 'lddt': 0.8514}
./data/pdb_chains/9B/9BA3.A.pdb ELBO_Y 0.03681946672549087 {'rmsd': 3.158, 'tm': 0.8826, 'gdt_ts': 0.8344, 'gdt_ha': 0.6625, 'lddt': 0.8509}
./data/pdb_chains/9B/9BA3.A.pdb ELBO_Y -0.02206151599332696 {'rmsd': 3.25, 'tm': 0.8806, 'gdt_ts': 0.8297, 'gdt_ha': 0.6438, 'lddt': 0.8494}
./data/pdb_chains/9B/9BA3.A.pdb ELBO_Y 0.0083573519418595 {'rmsd': 3.227, 'tm': 0.8782, 'gdt_ts': 0.8266, 'gdt_ha': 0.6562, 'lddt': 0.8468}
./data/pdb_chains/9B/9BA3.A.pdb ELBO_Y 0.0010533817919161559 {'rmsd': 3.295, 'tm': 0.8852, 'gdt_ts': 0.8453, 'gdt_ha': 0.6781, 'lddt': 0.8604}
"""

residue_data = {"7H07":170,"8RIN":584,"7HLW":188,"7GAU":121,"8X67":63,"5SN5":828,}
seqlen_data = {"7H07":170,"8RIN":292,"7HLW":188,"7GAU":121,"8X67":63,"5SN5":414,"4FPQ":131,"7EKA":121,"7FPL":159,"7FSL":195,"7FUW":149,"7G0O":135,"7GAD":121,"7GZ1":169,"7H08":170,"7H33":150,"7H6V":163,"7HGG":94,"7HGT":94,"7HIJ":163,"7HM1":188,"7N8G":135,"7Y86":89,"8B6E":66,"8R2Y":265,"8RSF":384,"8T68":274,"8UWM":279,"8VZG":260,"8X6E":310,"8XB1":346,"8XWG":320,"9BA3":288}

lines = input_data.strip().split("\n")
output_lines = ["chain,metric,rmsd,tm,gdt_ts,gdt_ha,lddt"]

for line in lines:
    parts = line.split(maxsplit=3)
    chain = parts[0].split("/")[-1].split(".")[0]  # Extract chain identifier
    metric = parts[2]
    metrics_data = eval(parts[3])  # Safely parse the dictionary in the string
    rmsd = metrics_data['rmsd']
    tm = metrics_data['tm']
    gdt_ts = metrics_data['gdt_ts']
    gdt_ha = metrics_data['gdt_ha']
    lddt = metrics_data['lddt']
    
    # Append formatted line
    output_lines.append(f"{chain},NAN,{seqlen_data[chain]},{metric},{rmsd},{tm},{gdt_ts},{gdt_ha},{lddt}")

# Print or save the output lines
for line in output_lines:
    print(line)