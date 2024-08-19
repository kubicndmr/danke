import os
import shutil
import argparse


def prefix(id, name = '', buffer = 5):
    return name + str(id).zfill(buffer)

def listdir(path, ending=None):
    '''Returns dir with full path'''
    if ending == None:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path) 
                       if f.endswith(ending)])

def get_files(target_path):
    files_csv = listdir(target_path, '.csv')
    files_txt = listdir(target_path, '.txt')
    
    # Check number of files
    assert len(files_csv) == len(files_txt), f"Existing number of csv and txt files dont match in {target_path}"
    
    # Check order of files and if they match
    match_flag = 1
    order_flag = 1
    unmatch_files = []
    unorder_files = []
    for i, (c,t) in enumerate(zip(files_csv, files_txt)): 
        match_flag *= c[:-4] == t[:-4]
        if not match_flag: unmatch_files.append([c,t])
        
        order_flag_csv = int(c[-9:-4]) == (i+1)
        order_flag_txt = int(t[-9:-4]) == (i+1)
        order_flag *= (order_flag_csv * order_flag_txt)
        if not order_flag_csv:unorder_files.append(c)
        if not order_flag_csv: unorder_files.append(t)
        
    assert match_flag == 1, f"following csv and txt file in {target_path} do not match {unmatch_files}"
    assert order_flag == 1, f"following files in {target_path} not ordered properly {unorder_files}"
    
    return files_csv, files_txt
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bumbdadumba")
        
    parser.add_argument('-m', '--mode', type=str,
                        default=None,
                        help='c for create, a for append')

    parser.add_argument('-f', '--from_path', type=str,
                        help='path to copy from')
        
    parser.add_argument('-t', '--to_path', type=str,
                        default=None,
                        help='path to copy')

    args = parser.parse_args()

    if args.mode == 'c':
        assert args.from_path != args.to_path, "Both paths cannot be same"
        assert not os.path.exists(args.to_path), f"Target path {args.to_path} already exists. Do you want to append instead?"
        
        os.mkdir(args.to_path)

        files_csv = listdir(args.from_path, '.csv')
        files_txt = listdir(args.from_path, '.txt')
        
        for i, (c,t) in enumerate(zip(files_csv, files_txt)):
            print(f"{c} \t-> \t{os.path.join(args.to_path, prefix(i+1, 'SynOP_'))+ '.csv'}")
            print(f"{t} \t-> \t{os.path.join(args.to_path, prefix(i+1, 'SynOP_'))+ '.txt'}")
            
            shutil.copy(c, os.path.join(args.to_path, prefix(i+1, 'SynOP_'))+ '.csv')
            shutil.copy(t, os.path.join(args.to_path, prefix(i+1, 'SynOP_'))+ '.txt')
            
    if args.mode == 'a':
        assert args.from_path != args.to_path, "Both paths cannot be same"
        
        # get file pairs
        to_files_csv, _ = get_files(args.to_path)
        from_files_csv, from_files_txt = get_files(args.from_path)
        
        last_index = int(to_files_csv[-1][-9:-4])
        input(f"Last index of existing files is {last_index}. Continue?")
        
        for f_i, t_i in enumerate(range(last_index+1, last_index+len(from_files_csv)+1)):
            print(f"{from_files_csv[f_i]} \t-> \t{os.path.join(args.to_path, prefix(t_i, 'SynOP_')+'.csv')}")
            shutil.copy(from_files_csv[f_i], os.path.join(args.to_path, prefix(t_i, 'SynOP_')+'.csv'))
            
            print(f"{from_files_txt[f_i]} \t-> \t{os.path.join(args.to_path, prefix(t_i, 'SynOP_')+'.txt')}")
            shutil.copy(from_files_txt[f_i], os.path.join(args.to_path, prefix(t_i, 'SynOP_')+'.txt'))