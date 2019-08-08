import sys

with open( sys.argv[1], 'r' ) as inF:
    with open( sys.argv[2], 'w' ) as outF:
        for line in inF:
            outF.write( ( line[:-1] + "|\n") ) 