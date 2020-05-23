#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''



'''


def parametrization(family, parametric, model):

    class_name = 'Parameters{0:s}{1:s}'.format(family.title(),
            parametric.title())

    parent_name = 'Parameters{0:s}'.format(family.title())
    raw_string = '''
class {0:s}({1:s}):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_{2:s}_{3:s}(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.{2:s}_{4:s}_{3:s}(t, a, tt, theta)

    '''

    return raw_string.format(class_name, parent_name, family, parametric, 
            model)

def wrap_python_reparametrize(family, parametric):
    ''' Generate the python code. '''

    code = '''
def reparametrize_{0:s}_{1:s}(theta):
    \'\'\' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    \'\'\'

    return wrapper(theta, c_wrappers.p_reparametrize_{0:s}_{1:s})
'''
    return code.format(family, parametric)
    

if __name__ == '__main__':
    model = ['two_states']
    family = ['seria']
    parametric = ['gamma', 'invgamma', 'mixedgamma', 'lognorm', 'later', 
        'wald']

    with open('./wrapped.tem', 'r') as fp:
        fstring = fp.read()

    with open('./likelihoods.tem', 'r') as fp:
        sllh = fp.read()

    with open('./reparametrize.tem', 'r') as fp:
        sreparam = fp.read()

    for f in family:
        for p in parametric:
            
            print((wrap_python_reparametrize(f, p)))

