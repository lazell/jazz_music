
from random import choice

'''
- - - - - - - - - - - - - - - - - - -
Jazz-Era Swing Band Name Generator
- - - - - - - - - - - - - - - - - - -

'''

# Data
first = '''Glenn Tommy Lionel Cab Bob Sidney Simone Nina Jack Crowder Guy Lillian Sarah Franky Benny
Lawrence Jack Gene Gertrude Peggy Fitzgerald Raymond Wynonie'''.split()

second = '''Miller Dorsey Hampton Kirk Day Goodman King Waller Godfrey Basie Estes Cole Brown Dizzy Monk Baker'''.split()

adj = '''Sleepy Royal Jelly-Roll Dixie New-Orleans Cotton Stomp Four Five Manhattan Swing Ink Jumping Smokey-Mountain
"The-Lion" Red Blue Twelve Cozy Southern Hot-Jazz Lockjaw'''.split()

title = '''Count Duke The-King The'''.split()

noun = '''Georgians Brothers Missourians Jazz-Band Jazz-Rhythm Gang Ensamble Feetwarmers Boys Sisters
Sextet Hot-Shots Coulds-of-Joy Stompers Back-Room-Boys'''.split()

noun2 = '''and-His-Orchestra and-His-Rhythym and-His-All-Stars Quintent Trio Dwarfs Skillet-Lickers'''.split()

def generate_swing_band_name(first,second,adj,noun,noun2):

    name_type1 = "{} {} {} {} {}".format(choice(first), choice(second),
                                         choice(title), choice(adj),
                                         choice(noun2))
    name_type2 = "The {} {} {}".format(choice(first), choice(second),
                                       choice(noun2))
    name_type3 = "{} {} and {} {} {}".format(choice(adj), choice(second),
                                             choice(title), choice(adj),
                                             choice(noun2))
    short_name1 = "{}'s {}".format(choice(first), choice(adj), choice(noun),
                                   choice(noun2))
    short_name2 = "{} {} {}".format(choice(first), choice(adj), choice(noun))
    short_name3 = "{} {} {}".format(choice(title), choice(adj), choice(noun))
    short_name4 = "{} {} {}".format(choice(adj), choice(title), choice(adj))

    model = [name_type1, name_type2, name_type3, short_name1, short_name2, short_name3, short_name4]
    print '----------------------------------------\n'
    print choice(model).replace("-", " ")
    print '\n----------------------------------------\n'

if __name__ == '__main__':
    for i in range(0,5):
        generate_swing_band_name(first,second,adj,noun,noun2), "\n"
