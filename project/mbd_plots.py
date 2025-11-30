import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os

'''
hese are the methods required to plot performance profiles.
every method has a descriptive docstring. 
'''

def parse_vector(vec_str):
    '''
    input: array of strings (filepaths)
    output information in the file in the form of pandas dataframe
    '''
    # remove brackets
    s = vec_str.strip()[1:-1].strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",")]


def parse(filePath):
    
    rows = []

    with open(filePath,"r") as fp:
        #jump header 
        next(fp)   

        for line in fp:
            
            #jump title rows if necessary
            if "k" in line and "f(x)" in line:
                
                continue   
            
            #drop empty line if necessary:
            if line.strip() == "":   
                continue

            #return an array for that row
            row = line.split("|")

            # jump incomplete rows
            if len(row) < 9:
                continue   

            #remove everything around the values so we read in just the value:
            k =int(row[0].strip())

            x = parse_vector(row[1].strip())

            fx = float(row[2].strip())
            
            #varibles
        
            delta = float(row[3].strip())
            target_acc = float(row[4].strip())
            
            
            norm_g = row[5].strip()

            #could alos just remove first row.
            g_norm = float(norm_g) if norm_g!= "-1.00" else None

            f_evals = int(row[6].strip())
            success = int(row[7].strip())

            msg = row[8].strip()

            rows.append({
                "k": k,
                
                "x": x,
                "f(x)": fx,
                "delta": delta,
                "target_acc": target_acc,
                "||g~||": g_norm,

                "f_evals": f_evals,
                "success": success,
                "msg": msg
            })

    return pd.DataFrame(rows)


def convPlot(df):
    '''
    input: dataframe
    output: convergence plot
    '''
    k = df['f_evals']
    fx = df['f(x)']
    plt.plot(k,fx)
    plt.xlabel("iteration count")
    plt.ylabel("best function value")
    plt.show()


def perfProf(str,f_star, tao,alphaFrom = 1, alphaTo = 10, alphaTicks = 100):
    '''
    takes in an array of strings (filepaths) and outputs
      a table containg all final rows of the file 
      (or the lowest F_evals where T_a,p = 1).
      the table is the performance profile data. 
      
      notes:
      f_star is the best value known to man'''
    dfs = []
    dfFinal = []
    for i in str:
        dfs.append(parse(i))
        #now a list of dataframes
    
    for i in dfs:
        #column for progress ratio: 
        i['fNacc'] = 1-((i['f(x)'] - f_star)/(i['f(x)'][0]-f_star))  
      
    #initialize list of final rows for problem instance per algo
    selected = []
    
    #need to find the row where fNacc> 1-tao if it exists:
    for df in dfs:
        #it doesnt exist
        if df.iloc[-1]['fNacc'] < 1-tao: 
            #fail :(
            selected.append(pd.DataFrame(df.iloc[-1]).T)
            
        else: #does exist
         
            #taking lowest f_evals that meets tolerance:
            selected.append(pd.DataFrame(df[(df['fNacc'] >= 1 - tao) ].iloc[0]).T)
        
    df = pd.concat(selected, ignore_index= True)
    
    #indicator (solution reaches. within tolerance
    df['T_ap'] = np.where(df['fNacc'] >= 1-tao ,1,0)
    
    #just renaming the vairable  
    df['N_ap'] = df['f_evals']
    N_ap_best = df.loc[df['T_ap'] == 1, 'N_ap'].min()
    
    #performance ratio
    df['r_ap'] = df['N_ap']/N_ap_best
    df.loc[df['T_ap'] == 0, 'r_ap'] = np.inf
    df = df.drop(columns = ['msg','delta','success','k','||g~||'])
    #performance ratio:
    alpha = np.linspace(alphaFrom, alphaTo, alphaTicks)
    rho_alpha = np.zeros(len(alpha))  
    countP = 0
    
    for i in range(len(alpha)):
        count = 0
        orderP = df.shape[0]
        for p in range(df.shape[0]):
            if df['r_ap'].iloc[p] <= alpha[i]:
                count = count+1
        rho_alpha[i] = count/ orderP
    return(pd.DataFrame(rho_alpha,alpha))

def find_log_paths(base_dir, log_name):

    '''
    input base directory (as string), file type (ex. .txt)
    output: list of filepaths that under that base directory
    
    '''
    target = log_name + ".txt"
    matches = []

    # walk the directory tree
    for root, dirs, files in os.walk(base_dir):
        if target in files:
            matches.append(os.path.join(root, target))
    
    return matches


def getBestHelper(str):
    '''
    just a helper method for get best( no need to read)
    '''
    
    dfs = []
    dfFinal = []
    for i in str:

        dfs.append(parse(i))
        #now a list of dataframe
        
    #initialize list of final rows for problem instance per algo
    selected = []
    
    #need to find the row where fNacc> 1-tao if it exists:
    for df in dfs:
        selected.append(pd.DataFrame(df.iloc[-1]).T)    
    df = pd.concat(selected, ignore_index= True)
    
    return(np.min(df['f(x)']))


def getBest(arr):
    
    '''
    inpput: 
    array of string ( base directory fileapths)
    output: the best f(x) value found accross all tables stored in those txt files
    
    '''
    d = np.zeros(len(arr))
    for i in range(len(arr)):
        d[i] = getBestHelper(arr[i])
    return(np.min(d))


def find_all_txt_paths(base_dir):
    ''' 
    input: base directory (as string) 
    output: all txt files under that base directory
    '''
    txt_paths = []

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".txt"):   # case-insensitive
                full_path = os.path.join(root, f)
                txt_paths.append(full_path)

    return txt_paths
