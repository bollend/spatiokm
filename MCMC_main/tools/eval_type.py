'''
Evaluate the type of a string
'''

def evaluate(string, string_type):

    if string_type=='FLOAT':

        try:
            
            return float(string)

        except ValueError as error:

            print('ValueError: The parameter \'%s\' of the object.dat file should be of type FLOAT' % (string))

    if string_type=='INT':

        try:

            return int(string)

        except ValueError as error:

            print('ValueError: The parameter \'%s\' of the object.dat file should be of type INT' % (string))

    if string_type=='STRING':

        try:

            return string

        except ValueError as error:

            print('ValueError: The parameter \'%s\' of the object.dat file should be of type STRING' % (string))


    if string_type=='BOOL':

        if string=='True':

            return True

        elif string=='False':

            return False

        else:

            print('ValueError: The parameter \'%s\' of the object.dat file should be of type BOOL' % (string))
