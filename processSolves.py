# This scripts validates the solves work and trims to the F2L stage

import pycuber as pc
from pycuber.solver.cfop.f2l import F2LSolver

outfile = open('processedSolves.txt','w')

with open('m3x3solves.txt') as infile:
    reconstruction = 0
    valid = 0
    invalid = 0
    for line in infile:
        line = line.strip() #strip \n at end of line
        solve = line.split('|')
        scramble = solve[0]
        solution = solve[1]
        solution = solution.split(' ')
        cube = pc.Cube()
        cube(scramble)
        f2l = F2LSolver(cube)
        solved_f2l = False

        n_move = 0
        for i in range(len(solution)):
            move = solution[i]
            if move != '/':
                cube(move)
                if f2l.is_solved():
                    solved_f2l = True
                    n_move = i
                    break

        if solved_f2l:
            f2l_solution = solution[:n_move+1]
            out_line = scramble + "|" + " ".join(f2l_solution)
            outfile.write(out_line + "\n")
            reconstruction += 1
            valid += 1
            print('Solve', reconstruction, '--> VALID.')
        else:
            reconstruction += 1
            invalid += 1
            #outfile.write("INVALIDATED" + "\n") # in case we want to know which solves were invalidated
            print('Solve', reconstruction, '--> INVALIDATED.')
    print(invalid, ' Invalid solves.')
    print(valid, ' Valid solves.')
