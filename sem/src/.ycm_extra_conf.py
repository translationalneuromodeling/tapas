##########################################################################
# Simple ycm_extra_conf.py example                                       #
# Copyright (C) <2013>  Onur Aslan  <onur@onur.im>                       #
#                                                                        #
# This file is loaded by default. Place your own .ycm_extra_conf.py to   #
# project root to override this.                                         #
#                                                                        #
# This program is free software: you can redistribute it and/or modify   #
# it under the terms of the GNU General Public License as published by   #
# the Free Software Foundation, either version 3 of the License, or      #
# (at your option) any later version.                                    #
#                                                                        #
# This program is distributed in the hope that it will be useful,        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of         #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
# GNU General Public License for more details.                           #
#                                                                        #
# You should have received a copy of the GNU General Public License      #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.  #
##########################################################################



# some default flags
# for more information install clang-3.2-doc package and
# check UsersManual.html
flags = [
'-Wall',
'-Werror',
'-std=c11',
# std is required
# clang won't know which language to use compiling headers

# '-x' and 'c++' also required
# use 'c' for C projects
'c',
'-I./',
'-I./antisaccades',
'-I/opt/local/include/',
'-I/usr/local/MATLAB/R2017a/extern/include',
'-I/Applications/MATLAB_R2017b.app/extern/include',
'-imath',
'-istdlib'
# include third party libraries
#'-isystem',
#'/usr/include/python2.7',
]

# youcompleteme is calling this function to get flags
# You can also set database for flags. Check: JSONCompilationDatabase.html in
# clang-3.2-doc package
def FlagsForFile( filename ):
  return {
    'flags': flags,
    'do_cache': True
  }
