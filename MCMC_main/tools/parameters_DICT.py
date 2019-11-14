'''
create the parameter dictionary
'''
import eval_type

def create_parameters(inputfile):
    with open(inputfile) as f:
        lines  = f.readlines()[2:]

    group=None
    par=None
    which_line = 'new'
    parameters= {}

    for l in lines:
        split_lines = l.split()

        if split_lines[0]=='FINISH':
            break

        if split_lines[0]=='GROUP':

            group = split_lines[1]
            parameters[group] = {}

        else:

            if group!='MODEL':

                parameters[group][split_lines[0]] = eval_type.evaluate(split_lines[1], split_lines[2])

            else:

               if which_line=='new':

                   parameters[group][ split_lines[0]] = {}
                   par = split_lines[0]
                   which_line = 'min'

               elif which_line=='min':

                   parameters[group][par]['min'] = eval_type.evaluate(split_lines[1], split_lines[2])
                   which_line = 'max'

               elif which_line=='max':

                   parameters[group][par]['max'] = eval_type.evaluate(split_lines[1], split_lines[2])
                   which_line = 'new'

    return parameters
