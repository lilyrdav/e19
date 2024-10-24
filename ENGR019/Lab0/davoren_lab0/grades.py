# Demonstrate very basic Python functionality (for loop, functions, if
# statements) by converting a list of numerical grades to letter
# grades.
#
######################################################################

# Define a function to convert a numeric grade to a letter
def number_to_lettergrade(gradenum):

    # The if statement checks a condition, and if it is met,
    # executes the code in the first indented block.
    if gradenum < 60:
        lettergrade = 'F'
    elif gradenum < 70: # elif -> "else if", checks another condition
        lettergrade = 'D'
    elif gradenum < 80: # etc.
        lettergrade = 'C'
    elif gradenum < 90: # etc.
        lettergrade = 'B'
    else: # "else" is the catch-all that executes if no condition met.
        lettergrade = 'A'

    # Note not all if statements need elif's and else's. You have to decide
    # if and when they are appropriate!

    # The "return" statement passes the result back to the caller of
    # the function (in this case, the main program below).
    return lettergrade

######################################################################
# Everything indented all the way to the left is the "main" program:

# Create a list of numeric grades
my_list = [35, 87, 95, 63]

# A for loop iterates over a collection (in this case, the list
# above), and replaces the loop variable (here, score) with each
# element of the list in turn.
for score in my_list:

    # Show the user the converted score here.
    print('a score of', score, 'is', number_to_lettergrade(score))

