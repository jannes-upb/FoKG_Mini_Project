import torch
import argparse
from pykeen.triples import TriplesFactory
from rdflib import Graph, RDF

def main(input_file, output_file):
    # Load the reference triples using PyKEEN
    reference_data_pykeen = TriplesFactory.from_path_binary("trans-e-embeddings/training_triples")

    # Load the pre-trained TransE model
    model = torch.load("trans-e-embeddings/trained_model.pkl", map_location=torch.device('cpu'))

    # Set the device to use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the architecture of the classifier
    classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=150, out_features=300),  # First layer: 150 -> 300
        torch.nn.ReLU(),                                   # Activation function
        torch.nn.Linear(in_features=300, out_features=100), # Second layer: 300 -> 100
        torch.nn.ReLU(),                                   # Activation function
        torch.nn.Linear(in_features=100, out_features=1),   # Final layer: 100 -> 1
    ).to(device)

    # Load the classifier model parameters from the saved file
    classifier.load_state_dict(torch.load("classifier/fact_checking_model.pt", map_location=torch.device("cpu")))

    # Load the test RDF graph
    test_graph = Graph()
    test_graph.parse(input_file)

    # Initialize lists to store IRIs, processed triples, and labels
    test_triple_iris = []
    test_triples = []

    # Process the test graph to extract triples and convert them into tensor representations
    for statement in test_graph.subjects(RDF.type, RDF.Statement):
        # Extract subject, predicate, and object from the RDF statement
        subject = test_graph.value(statement, RDF.subject)
        predicate = test_graph.value(statement, RDF.predicate)
        obj = test_graph.value(statement, RDF.object)

        # Convert the subject, predicate, and object to their respective IDs
        subject_id = reference_data_pykeen.entity_to_id[str(subject)]
        predicate_id = reference_data_pykeen.relation_to_id[str(predicate)]
        obj_id = reference_data_pykeen.entity_to_id[str(obj)]

        # Obtain tensor representations for subject, predicate, and object
        subject_tensor = model.entity_representations[0](torch.LongTensor([subject_id]))
        predicate_tensor = model.relation_representations[0](torch.LongTensor([predicate_id]))
        obj_tensor = model.entity_representations[0](torch.LongTensor([obj_id]))

        # Store the statement IRI and concatenated tensor representation
        test_triple_iris.append(statement)
        test_triples.append(torch.cat((subject_tensor, predicate_tensor, obj_tensor), dim=1))

    # Stack the test triples into a single tensor
    x_test = torch.stack(test_triples)

    # Set the classifier to evaluation mode
    classifier.eval()

    # Perform predictions on the test triples
    test_logits = classifier(x_test).squeeze()  # Get logits from the classifier
    test_pred = torch.sigmoid(test_logits)  # Apply sigmoid

    # Convert predictions to a list of values
    test_pred_as_list = [x.item() for x in list(test_pred)]

    # Write the predictions to a results file in TTL format
    with open(output_file, "w") as result_file:
        for i in range(len(test_pred_as_list)):
            result_file.write(
                f"<{test_triple_iris[i]}> <http://swc2017.aksw.org/hasTruthValue> \"{test_pred_as_list[i]}\"^^<http://www.w3.org/2001/XMLSchema#double> .\n"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the classifier on a RDF graph.")
    parser.add_argument("input_file", type=str, help="Path to the input RDF file.")
    parser.add_argument("--output_file", type=str, default="predictions.ttl",
                        help="Path to the output TTL file (default: predictions.ttl).")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
