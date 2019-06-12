# -*- coding: utf-8 -*-
"""
Copyright (c) 2019, Yifan Peng
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import logging
import string

import sympy

ACCENTS = {
    u'ά': u'a', u'Ά': u'Α',
    u'έ': u'e', u'Έ': u'Ε',
    u'ή': u'h', u'Ή': u'H',
    u'ί': u'e', u'Ί': u'Ι',
    u'ύ': u'u', u'Ύ': u'Y',
    u'ό': u'o', u'Ό': u'O',
    u'ώ': u'w', u'Ώ': u'w',
    u'Ã': u'A', u'Å': u'A',
    u'ç': u'c', u'ï': 'i',
}

# The possible string conversions for each case.
GREEK_CONVERT_STRINGS = {
    u"αι": [u"ai", u"e"],
    u"Αι": [u"Ai", u"E"],
    u"ΑΙ": [u"AI", u"E"],
    u"ει": [u"ei", u"i"],
    u"Ει": [u"Ei", u"I"],
    u"ΕΙ": [u"EI", u"I"],
    u"οι": [u"oi", u"i"],
    u"Οι": [u"Oi", u"I"],
    u"ΟΙ": [u"OI", u"I"],
    u"ου": [u"ou", u"oy", u"u"],
    u"Ου": [u"Ou", u"Oy", u"U"],
    u"ΟΥ": [u"OU", u"OY", u"U"],
    u"ευ": [u"eu", u"ef", u"ev", u"ey"],
    u"Ευ": [u"Eu", u"Ef", u"Ev", u"Ey"],
    u"ΕΥ": [u"EU", u"EF", u"EV", u"EY"],
    u"αυ": [u"au", u"af", u"av", u"ay"],
    u"Αυ": [u"Au", u"Af", u"Av", u"Ay"],
    u"ΑΥ": [u"AU", u"AF", u"av", u"AY"],
    u"μπ": [u"mp", u"b"],
    u"Μπ": [u"Mp", u"B"],
    u"ΜΠ": [u"MP", u"B"],
    u"γγ": [u"gg", u"g"],
    u"Γγ": [u"Gg", u"G"],
    u"ΓΓ": [u"GG", u"G"],
    u"γκ": [u"gk", u"g"],
    u"Γκ": [u"Gk", u"G"],
    u"ΓΚ": [u"GK", u"G"],
    u"ντ": [u"nt", u"d"],
    u"Ντ": [u"Nt", u"D"],
    u"ΝΤ": [u"NT", u"D"],
    u"α": [u"a"],
    u"Α": [u"A"],
    u"β": [u"b", u"v"],
    u"Β": [u"B", u"V"],
    u"γ": [u"g"],
    u"Γ": [u"G"],
    u"δ": [u"d"],
    u"Δ": [u"D"],
    u"ε": [u"e"],
    u"Ε": [u"E"],
    u"ζ": [u"z"],
    u"Ζ": [u"Z"],
    u"η": [u"h", u"i"],
    u"Η": [u"H", u"I"],
    u"θ": [u"th", u"8"],
    u"Θ": [u"TH", u"8"],
    u"ι": [u"i"],
    u"Ι": [u"I"],
    u"κ": [u"k"],
    u"Κ": [u"K"],
    u"λ": [u"l"],
    u"Λ": [u"L"],
    u"μ": [u"m"],
    u"Μ": [u"M"],
    u"ν": [u"n"],
    u"Ν": [u"N"],
    u"ξ": [u"x", u"ks"],
    u"Ξ": [u"X", u"KS"],
    u"ο": [u"o"],
    u"Ο": [u"O"],
    u"π": [u"p"],
    u"Π": [u"P"],
    u"ρ": [u"r"],
    u"Ρ": [u"R"],
    u"σ": [u"s"],
    u"Σ": [u"S"],
    u"ς": [u"s"],
    u"τ": [u"t"],
    u"Τ": [u"T"],
    u"υ": [u"y", u"u", u"i"],
    u"Υ": [u"Y", u"U", u"I"],
    u"φ": [u"f", u"ph"],
    u"Φ": [u"F", u"PH"],
    u"χ": [u"x", u"h", u"ch"],
    u"Χ": [u"X", u"H", u"CH"],
    u"ψ": [u"ps"],
    u"Ψ": [u"PS"],
    u"ω": [u"w", u"o", u"v"],
    u"Ω": [u"w", u"O", u"V"],
}

OTHERS = {
    u'\xb7': '*',  # MIDDLE DOT
    u'\xb1': '+',  # PLUS-MINUS SIGN
    u'\xae': 'r',  # REGISTERED SIGN
    u'\u2002': ' ',  # EN SPACE
    u'\xa9': 'c',  # COPYRIGHT SIGN
    u'\xa0': ' ',  # NO-BREAK SPACE
    u'\u2009': ' ',  # THIN SPACE
    u'\u025b': 'e',  # LATIN SMALL LETTER OPEN E
    u'\u0303': '~',  # COMBINING TILDE
    u'\u043a': 'k',  # CYRILLIC SMALL LETTER KA
    u'\u2005': ' ',  # FOUR-PER-EM SPACE
    u'\u200a': ' ',  # HAIR SPACE
    u'\u2026': '.',  # HORIZONTAL ELLIPSIS
    u'\u2033': '"',  # DOUBLE PRIME
    u'\u2034': '"',  # TRIPLE PRIME
    u'\u2075': '5',  # SUPERSCRIPT FIVE
    u'\u2077': '7',  # SUPERSCRIPT SEVEN
    u'\u2079': '9',  # SUPERSCRIPT NINE
    u'\u207a': '+',  # SUPERSCRIPT PLUS SIGN
    u'\u207b': '-',  # SUPERSCRIPT MINUS
    u'\u2080': '0',  # SUBSCRIPT ZERO
    u'\u2081': '1',  # SUBSCRIPT ONE
    u'\u2082': '2',  # SUBSCRIPT TWO
    u'\u2083': '3',  # SUBSCRIPT THREE
    u'\u2084': '4',  # SUBSCRIPT FOUR
    u'\u2085': '5',  # SUBSCRIPT FIVE
    u'\u2122': 'T',  # TRADE MARK SIGN
    u'\u2192': '>',  # RIGHTWARDS ARROW
    u'\u2217': '*',  # STERISK OPERATOR
    u'\u223c': '~',  # TILDE OPERATOR
    u'\u2248': '=',  # ALMOST EQUAL TO
    u'\u2264': '<',  # LESS-THAN OR EQUAL TO
    u'\u2265': '>',  # GREATER-THAN OR EQUAL TO
    u'\u22c5': '*',  # DOT OPERATOR
    u'\ue232': 'x',  #
    u'\ue2f6': 'x',  # Chinese character
    u'\xb0': '*',  # DEGREE SIGN
    u'\xb2': '2',  # SUPERSCRIPT TWO
    u'\xb3': '3',  # SUPERSCRIPT THREE
    u'\xb4': '\'',  # ACUTE ACCENT
    u'\xb5': 'm',  # MICRO SIGN
    u'\xb9': '1',  # SUPERSCRIPT ONE
    u'\xc3': 'A',  # LATIN CAPITAL LETTER A WITH TILDE
    u'\xc5': 'A',  # LATIN CAPITAL LETTER A WITH RING ABOVE
    u'\xd7': '*',  # MULTIPLICATION SIGN
    u'\xe7': 'c',  # LATIN SMALL LETTER C WITH CEDILLA
    u'\xef': 'i',  # LATIN SMALL LETTER I WITH DIAERESIS
    u'\xf8': 'm',  # LATIN SMALL LETTER O WITH STROKE
    u'\xfc': 'u',  # LATIN SMALL LETTER U WITH DIAERESIS
    u'\xf6': 'o',  # LATIN SMALL LETTER O WITH DIAERESIS
    u'\u2194': '<',  # LEFT RIGHT ARROW
    u'\xe1': 'a',  # LATIN SMALL LETTER A WITH ACUTE
    u'\u221e': '~',  # INFINITY
    u'\u2193': '<',  # DOWNWARDS ARROW
    u'\u2022': '*',  # BULLET
    u'\u2211': 'E',  # N-ARY SUMMATION
    u'\xdf': 'b',  # LATIN SMALL LETTER SHARP S
    u'\xff': 'y',  # LATIN SMALL LETTER Y WITH DIAERESIS
    u'\u2550': '=',  # BOX DRAWINGS DOUBLE HORIZONTAL
    u'\u208b': '-',  # SUBSCRIPT MINUS
    u'\u226b': '>',  # MUCH GREATER-THAN
    u'\u2a7e': '>',  # GREATER-THAN OR SLANTED EQUAL TO
    u'\uf8ff': '*',  # Private Use, Last
    u'\xe9': 'e',  # LATIN SMALL LETTER E WITH ACUTE
    u'\u0192': 'f',  # LATIN SMALL LETTER F WITH HOOK
    u'\u3008': '(',  # LEFT ANGLE BRACKET
    u'\u3009': ')',  # RIGHT ANGLE BRACKET
    u'\u0153': 'o',  # LATIN SMALL LIGATURE OE
    u'\u2a7d': '<',  # LESS-THAN OR SLANTED EQUAL TO
    u'\u2243': '=',  # ASYMPTOTICALLY EQUAL TO
    u'\u226a': '<',  # much less-than
}


def printable(s: str, greeklish=False, verbose=False, replacement=' ') -> str:
    """
    Return string of ASCII string which is considered printable.
    """
    out = ''
    for c in s:
        if c in string.printable:
            out += c
        else:
            if greeklish:
                if c in ACCENTS:
                    out += ACCENTS[c]
                elif c in GREEK_CONVERT_STRINGS:
                    out += GREEK_CONVERT_STRINGS[c][0]
                elif c in OTHERS:
                    out += OTHERS[c]
                elif verbose:
                    logging.warning('Unknown char: %r', sympy.pretty(c))
                    out += replacement
            else:
                if verbose:
                    logging.warning('Cannot convert char: %s', c)
                out += replacement
    return out
