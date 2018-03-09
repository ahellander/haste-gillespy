""" Example of usage, high-level API. """
import numpy 
import time
import datetime
import json

# Define gillespy model
import gillespy
class GRNPopulation(gillespy.Model):

    def __init__(self, mu=10.0,kappa=10.0,ka=1e7,kd=0.01,gamma_m=0.02, gamma_p=0.02):

        # Initialize the model.
        gillespy.Model.__init__(self, name="GRN")

        # Parameters
        NA = 6.022e23
        V  = 37e-15 
        
        mu = gillespy.Parameter(name='mu', expression=mu)
        kappa = gillespy.Parameter(name='kappa', expression=kappa)
        ka = gillespy.Parameter(name='ka', expression=ka/(NA*V))
        kd = gillespy.Parameter(name='kd', expression=kd)
        gamma_m = gillespy.Parameter(name='gamma_m', expression=gamma_m)
        gamma_p = gillespy.Parameter(name='gamma_p', expression=gamma_p)

        self.add_parameter([mu,kappa,ka,kd,gamma_m,gamma_p])

        # Species
        G_f = gillespy.Species(name='G_f', initial_value=1)
        G_o = gillespy.Species(name='G_o', initial_value=0)
        mRNA = gillespy.Species(name='mRNA', initial_value=0)
        P = gillespy.Species(name='P', initial_value=0)

        self.add_species([G_f,G_o,mRNA,P])

        # Reactions
        rxn1 = gillespy.Reaction(name = 'R1',reactants={G_f:1,P:1}, products = {G_o:1}, rate=ka)
        rxn2 = gillespy.Reaction(name = 'R2',reactants={mRNA:1}, products = {mRNA:1,P:1}, rate=kappa)
        rxn3 = gillespy.Reaction(name = 'R3',reactants={G_f:1}, products = {G_f:1,mRNA:1},rate=mu)
        rxn4 = gillespy.Reaction(name = 'R4',reactants={mRNA:1}, products = {}, rate=gamma_m)
        rxn5 = gillespy.Reaction(name = 'R5',reactants={P:1}, products = {}, rate=gamma_p)
        rxn6 = gillespy.Reaction(name = 'R6',reactants={G_o:1}, products = {G_f:1,P:1}, rate=kd)

        self.add_reaction([rxn1,rxn2,rxn3,rxn4,rxn5,rxn6])
        self.tspan=numpy.linspace(0,3600,1000)              


# Simple feature extraction
def extract_features(result):
    suma = numpy.sum(result,axis=0)
    maxa = numpy.max(result,axis=0)
    mina = numpy.min(result,axis=0)
    mean = numpy.mean(result,axis=0)
    features = {"sum":suma,"max":maxa, "min":mina,"mean":mean}
    return features


# Toy-example for now, we have massive parallel code to simulate
def generate_simulation_data(model):
    result = model.run()
    #A = result.get_species("A")
    features = extract_features(result[0][:,1::])
    #speca = numpy.array2string(A)
    return result, features
        
    
if __name__ == "__main__":
    model = GRNPopulation()
    result, features  = generate_simulation_data(model)
    print(features)

    
# The storage client will be used to handle data
#sc = HasteStorageClient()

# Create an experiment 
# name should probably be unique, and map directly to the "root stream id"
#E = haste.Experiment(storage_client=sc)

# Add a time series to the experiment
#ts = haste.TimeSeries()

# By adding the ts to the experiment, it will be assigned a unique substream-id, 
# linked somehow to the root stream id of the experiment. 
#E.add(ts)

# Now add (time, spatial_data_frame) tuples to the timeseries.
# They should be automatically handled by the storage client (i.e. passed through feature exraction, classification and policy-evaluation) 
#tspan = numpy.linspace(0,10.0,100)
#for t in tspan:
#    data = 100*["Large spatial dataframe goes here."]
#    ts.append((t,data))



