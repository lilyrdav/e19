import numpy as np

# Declare some 16-bit floating-point numbers
a = np.float16(128)
c = np.float16(16)
ep = np.float16(0.2)

# Add them together and print them out:

print('128 + 0.2 =',a+ep)
print('128 + 0.2 + 0.2 =',a+ep+ep)
print('128 + 0.2 + 0.2 + 0.2 =',a+ep+ep+ep)
print('128 + (0.2 + 0.2 + 0.2) =',a+(ep+ep+ep))

print('')

print('16 + 0.2 =',c+ep)
print('16 + 0.2 + 0.2 =',c+ep+ep)
print('16 + 0.2 + 0.2 + 0.2 =',c+ep+ep+ep)
print('16 + (0.2 + 0.2 + 0.2) =',c+(ep+ep+ep))
