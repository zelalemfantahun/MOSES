__author__ = 'eyob'
# Tested on python3.4

import time
import subprocess
import sys
import pathlib as path





class experimentation:


    def __init__(self):



        # self.file = '/home/opencog/eddy/docs/Hershey/time-series algo/binarized/2012-11-04 2013-01-05=UP=train.csv'
        self.file = '/home/zelalem/Downloads/moses_input/2012-11-04-2013-01-05=UP=train.csv'
        # self.file = '/home/opencog/eddy/docs/Hershey/time-series algo/binarized/2012-11-04 2013-01-05=UP=train2.csv'



        self.INPUT_FILE = ['-i','']
        self.OUTPUT_WITH_LABELS  = ['-W1','']
        self.OUTPUT_EVAL_NUMBER = ['-V1','']
        self.OUTPUT_FILE = ['-o','']
        self.BALANCING = ['--balance=1','']
        self.JOBS = ['-j8','-j7','-j6','']
        self.NUMBER_OF_EVALUATIONS = ['-m5000000','-m1000000','-m50000','-m100000','-m5','-m10000','']
        self.INTEGRATED_FEATURE_SELECTION = ['--enable-fs=1','']
        self.FEATURE_COUNT = ['--fs-target-size=20','']
        self.RESULT_COUNT = ['--result-count=100','--result-count=-1','--result-count=50', '--result-count=10','']
        self.MAX_CANDIDATES=['--max-candidates-per-deme=10','']
        self.DIVERSITY_PENALTY = ['--diversity-pressure=20','']
        self.OUTPUT_CSCORE = ['--output-cscore=1','']
        self.RANDOM_SEED = ['--random-seed=4582','']
        self.LOG_LEVEL = ['--log-level=DEBUG','']
        self.COMPLEXITY_RATIO = ['-z0.5','']
        self.FITNESS_FUNCTION = ['-Hpre','-Hrecall','f_one','']
        self.q = ['-q0.01','-q0.05','-q0.10','-q0.93','']
        self.w = ['-w0.1','-w0.15','-w0.2','']
        self.COMBO_PROGRAM = ['-C','-c','']
        self.TARGET_FEATURE = ['-ucase','']
        # print (self.TARGET_FEATURE)
        self.scoring = ['which_ever_is_greater','percent']
        self.scoring_percent = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0,0.0001]
        self.scoring_value = self.scoring[1]
        self.scoring_percent_value = 1



    def initialization(self):

        self.file_test = self.file[:-9]+'test.csv'
        self.index_of_forward_slashes = [i for i, letter in enumerate(self.file) if letter == '/']
        self.director_of_input_file = self.file[:self.index_of_forward_slashes[-1]+1]
        self.file_name_moses = self.file[self.index_of_forward_slashes[-1]+1:-4]
        self.combo_file_name = self.director_of_input_file + 'combos/' + self.file_name_moses+'=combos'
        self.log_file_name =  '-f'+self.director_of_input_file + 'log/' + self.file_name_moses+'=moses.log'
        self.log_file_name_eval_table =  '-f'+self.director_of_input_file + 'log/' + self.file_name_moses+'=eval-table.log'
        self.eval_out_file_name =  self.director_of_input_file + 'out/Out-' + self.file_name_moses+'='
        self.results_file_name =  self.director_of_input_file + 'results/' + self.file_name_moses[:-6]+'=Ones and zeros'
        self.evaluation_file_name =  self.director_of_input_file + 'results/' + self.file_name_moses[:-6]+'=evaluation'




    def run_moses(self):

        print('In "run_moses" method')
        print(time.strftime("%c")+'\n')
        # You can start commenting here
        print("Started training with moses")
        self.start_time = time.time()
        # Candidate moses options 2, no diverse models
        return_code = subprocess.call(['moses',self.INPUT_FILE[0],self.file,self.OUTPUT_WITH_LABELS[0],self.OUTPUT_EVAL_NUMBER[0],self.OUTPUT_FILE[0],self.combo_file_name,self.BALANCING[-1],self.JOBS[0],self.NUMBER_OF_EVALUATIONS[1],self.RESULT_COUNT[0],self.INTEGRATED_FEATURE_SELECTION[-1],self.FEATURE_COUNT[-1],self.FITNESS_FUNCTION[0],self.q[1],self.w[0],self.log_file_name,self.OUTPUT_CSCORE[-1],self.RANDOM_SEED[-1],self.LOG_LEVEL[-1],self.COMPLEXITY_RATIO[-1]])

        if return_code != 0:
            print("Failed to execute moses.")
            sys.exit(return_code)  # Is return_code the right value to give?
        self.end_time = time.time()
        print ('Training took ' + str(int((self.end_time - self.start_time)/60)) + ' minutes.')

        # You can comment upto this

    def cleaned_combo_generation(self):

        with open(self.combo_file_name, newline='\n') as f:
            lines = f.read().splitlines()

        print('-------Printing combo lines--------')
        self.combo_file_name_2 = self.combo_file_name+'-cleaned'
        f = open(self.combo_file_name_2, 'w')

        for i in range(len(lines)-1):
            # print(lines[i])
            first_space_character = lines[i].find(' ')
            self.combo_program = lines[i][first_space_character+1:]
            # print('combo program ' + self.combo_program)
            # print('First space character starts at ' + str(first_space_character))
            f.write(self.combo_program)
            f.write('\n')
        f.close()


    def run_eval_table(self):
        print(self.eval_out_file_name)
        return_code = subprocess.call(['eval-table',self.INPUT_FILE[0],self.file_test, self.COMBO_PROGRAM[0], self.combo_file_name_2,self.OUTPUT_FILE[0],self.eval_out_file_name,self.TARGET_FEATURE[0],self.log_file_name_eval_table])

        if return_code != 0:
            print("Failed to execute eval-table.")
            sys.exit(return_code)  # Is return_code the right value to give?

        # Do the voting here
    def populate_score_matrix(self):

        p = path.Path(self.director_of_input_file + 'out')
        # print(self.director_of_input_file + 'out')
        # print('Out-' + self.file_name_moses+'=[0-9]*')
        files = list(p.glob('Out-' + self.file_name_moses+'=[0-9]*'))
        self.number_of_files = len(files)

        print('number of files=',self.number_of_files)

        with open(str(files[0]), newline='\n') as f:
            lines3 = f.read().splitlines()

        # # print('lines3',lines3)
        self.number_of_scores = len(lines3)-1
        print(self.number_of_files,self.number_of_scores)

        self.scores = [[0 for x in range(self.number_of_files+1)] for x in range(self.number_of_scores)]
        # self.print_two_d_array(scores)

        f = ''

        for i in range(self.number_of_files):

            f = open(str(files[i]))

            value = f.readline()

            for j in range(self.number_of_scores):
                self.scores[j][i] = int(f.readline())

            f.close()
            # os.remove(str(files[i]))

    def sum_scores_matrix(self):

        sum = 0
        zeros = 0

        # Number of files means number of the different combo programs used to score the feature vectors
        # Number of scores is the number of rows. A column is the score given to all feature vectors in the test_set by the combo program associated with that column
        print('number of scores=',self.number_of_scores)
        print('number of files=',self.number_of_files)

        # self.print_two_d_array(scores)

        f1 = open(self.results_file_name, 'w')
        f1.write(self.file_name_moses[:-6]+'=Ones and zeros'+'\n')



        for i in range(self.number_of_scores):
            sum = 0
            for j in range(self.number_of_files):
                sum += self.scores[i][j]

            zeros = self.number_of_files - sum

            self.scores[i][self.number_of_files] = self.score(sum=sum,zeros=zeros,total_count=self.number_of_files)

             # print('Ones='+str(sum)+','+'Zeros='+str(zeros))
            f1.write(str(i)+' Ones='+str(sum)+','+'Zeros='+str(zeros)+'\n')

        f1.close()

        # self.print_two_d_array(self.scores)

        f = open(self.results_file_name+'=voted', 'w')
        f.write('OUT\n')
        for i in range(len(self.scores)):
            f.write(str(self.scores[i][self.number_of_files]))
            f.write('\n')
        f.close()

    def read_scores(self):

        self.lines2 = []

        for i in range(len(self.scores)):
            self.lines2.append(self.scores[i][self.number_of_files])

        # print(lines2)

        self.size = self.number_of_scores
        print('size =',self.size)

        with open(self.file_test, newline='\n') as f:
            self.lines4 = f.read().splitlines()

        self.lines4 = self.lines4[1:]

        self.out_of_test_set = []

        for i in range(len(self.lines4)):
            # print(lines4[i])
            # print(lines4[i].split(',')[0])
            self.out_of_test_set.append(int(self.lines4[i].split(',')[0]))

        # print(out_of_test_set)

    def generate_stats(self):

        self.number_of_ones_in_test_set = self.out_of_test_set.count(1)
        self.true_positive = self.false_positive = self.true_negative = self.false_negative = 0


        for k in range(self.size):
            if(self.out_of_test_set[k] == 0):
                if(int(self.lines2[k]) == 0):
                    self.true_negative += 1
                else:
                    self.false_positive += 1
            else:
                if(int(self.lines2[k]) == 1):
                    self.true_positive += 1
                else:
                    self.false_negative += 1



        if self.size != 0:
            accuracy = round(((self.true_negative +self.true_positive) / self.size)*100,2)
        else:
            accuracy = -1
        if  self.number_of_ones_in_test_set !=0:
            recall = round((self.true_positive / self.number_of_ones_in_test_set)*100,2)
        else:
            recall = -1
        if (self.true_positive + self.false_positive) != 0:
            precision = round((self.true_positive / (self.true_positive + self.false_positive))*100,2)
        else:
            precision = -1

        # print(self.file_name_moses[:-6])

        print('For:' + self.file_name_moses[:-6])
        print('Training took ' + str(int((self.end_time - self.start_time)/60)) + ' minutes.'+'\n')
        print('combo program ' + self.combo_program+'\n')
        print('size = ' + str(self.size)+'\n')
        print('Classification Accuracy = ' + str(accuracy)+'%')
        print('Recall = ' + str(recall)+'%')
        print('Precision = ' + str(precision)+'%')
        print('True positives (recall) = ' + str(self.true_positive) + '/' + str(self.number_of_ones_in_test_set))
        print('False positives = ' + str(self.false_positive) + '/' + str(self.size - self.number_of_ones_in_test_set))
        print('True negatives = ' + str(self.true_negative) + '/' + str(self.size - self.number_of_ones_in_test_set))
        print('False negatives = ' + str(self.false_negative) + '/' + str(self.number_of_ones_in_test_set))

        # start_time = end_time = 0


        with open(self.evaluation_file_name, 'w') as f:
            f.write(time.strftime("%c")+'\n')
            f.write('For:' + self.file_name_moses[:-6]+'\n')
            f.write('Training took ' + str(int((self.end_time - self.start_time)/60)) + ' minutes.'+'\n')
            f.write('combo program ' + self.combo_program+'\n')
            f.write('size = ' + str(self.size)+'\n')
            f.write('Classification Accuracy = ' + str(accuracy)+'%'+'\n')
            f.write('Recall = ' + str(recall)+'%'+'\n')
            f.write('Precision = ' + str(precision)+'%'+'\n')
            f.write('True positives (recall) = ' + str(self.true_positive) + '/' + str(self.number_of_ones_in_test_set)+'\n')
            f.write('False positives = ' + str(self.false_positive) + '/' + str(self.size - self.number_of_ones_in_test_set)+'\n')
            f.write('True negatives = ' + str(self.true_negative) + '/' + str(self.size - self.number_of_ones_in_test_set)+'\n')
            f.write('False negatives = ' + str(self.false_negative) + '/' + str(self.number_of_ones_in_test_set)+'\n')



    # This method tries to score a single row from the test dataset
    def score(self,sum, zeros,total_count):

        # print('sum =',sum,',zeros =',zeros,',total_count =',total_count,'sum/total_count =',sum/total_count)

        if(self.scoring_value == self.scoring[0]):
            if sum >= zeros:
                return 1
            else:
                return 0

        if(self.scoring_value == self.scoring[1]):
            # print('Using ',self.scoring_percent[self.scoring_percent_value]*100, '% of ones')
            if (sum/total_count) >= self.scoring_percent[self.scoring_percent_value]:
                return 1
            else:
                return 0

    def run_experimentation(self):

        self.initialization()
        self.run_moses()
        self.cleaned_combo_generation()
        self.run_eval_table()
        self.populate_score_matrix()
        self.sum_scores_matrix()
        self.read_scores()
        self.generate_stats()




    # ===============================================================
    # Assorted utilities and accessor functions
    # ===============================================================

    # Takes input a two dimenstional array and prints it
    def print_two_d_array(self,two_d_array):

        print('Matrix dimensions:'+str(len(two_d_array))+'x'+str(len(two_d_array[0])))
        for i in range(len(two_d_array)):
            print(i,':',two_d_array[i])
            pass
        print('Matrix dimensions:'+str(len(two_d_array))+'x'+str(len(two_d_array[0])))






if __name__ == '__main__':


    a = experimentation()
    a.run_experimentation()


