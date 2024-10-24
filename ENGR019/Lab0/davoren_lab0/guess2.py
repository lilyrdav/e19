# guess2.py
import numpy

# Display a message and wait for Enter
input('Think of a number between 1 and 100 then hit Enter...')

lowest = 1
highest = 100

# Loop forever
for i in range(7):

    print(f'You are on guess {i+1}')

    # Make a random guess between 1 and 100
    myguess = numpy.random.randint(lowest, highest + 1)

    # Ask the user if correct
    answer = input(f'Is it {myguess}?')

    if answer == 'c':
        print('I win!')
        break

    elif answer == 'h':
        #Only guess a number above myguess but lower than the highest number that has already been guessed
        lowest = myguess + 1

    elif answer == 'l':
        #Only guess a number below myguess but higher than the lowest number that has already been guessed
        highest = myguess - 1
    
