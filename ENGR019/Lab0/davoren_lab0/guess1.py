# guess1.py

import numpy

# Pick a random integer between 1 and 100.
num = numpy.random.randint(100)+1

# Give the user a big hint
print(f'Hint: you should guess {num}')

# Loop forever:
for i in range(7):

    print(f'You are on guess {i+1}')

    # Get the input as a string of characters
    userstr = input('Guess a number between 1 and 100: ')

    # Try to convert it to an integer
    try:

        userint = int(userstr)

        # Conversion was successful, check if correct
        if userint == num:

            print('Correct!')
            break

        elif userint < num:

            print('Too low!')

        elif userint > num:

            print('Too high!')
            

    except:

        # Not successful
        print('That wasn\'t a number.')