"""
you need to download the package and install it yourself.
follow the instructions from this link:
http://clic.cimec.unitn.it/composes/toolkit/installation.html
after you downloaded the git folder and before instalation,
you need to run 2to3 script in the main folder with the flag "w" so the code will be compatable
"""

from composes.semantic_space.space import Space

words_space = Space.build(data="sentence.out",
    rows="words.out", cols="words.out", format="sm")

#TODO: finish the assignmet. this package does *EVERYTHING* we need