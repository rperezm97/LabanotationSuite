import os
from owlready2 import *

# Function to load an existing ontology or create a new one with necessary classes and properties
def load_or_create_ontology(path="laban.owl"):
    if os.path.exists(path):
        # Load existing ontology from file
        onto = get_ontology(f"file://{os.path.abspath(path)}").load()
    else:
        # Create a new ontology if the file doesn't exist
        onto = get_ontology(f"file://{os.path.abspath(path)}")
        with onto:
            # --- Fundamental Classes ---
            class Symbol(Thing): pass
            class BasicSymbol(Symbol): pass
            class EffortSymbol(Symbol): pass
            
            class Action(Thing): pass

            class Staff(Thing): pass

            
            # --- Object and Data Properties ---
            class hasSymbol(ObjectProperty):
                domain = [Action, Staff]
                range = [Symbol]

            class hasJointType(DataProperty, FunctionalProperty):
                domain = [Symbol]
                range = [str]

            class hasStartTime(DataProperty, FunctionalProperty):
                domain = [Symbol, Action, Staff]
                range = [float]

            class hasEndTime(DataProperty, FunctionalProperty):
                domain = [Symbol, Action, Staff]
                range = [float]

            class isSimultaneousWith(ObjectProperty):
                domain = [Symbol]
                range = [Symbol]
                symmetric = True

            class previousSymbol(ObjectProperty, FunctionalProperty):
                domain = [Symbol]
                range = [Symbol]

            class nextSymbol(ObjectProperty, FunctionalProperty):
                domain = [Symbol]
                range = [Symbol]

            class isStartSymbol(DataProperty, FunctionalProperty):
                domain = [Symbol]
                range = [bool]

            class isEndSymbol(DataProperty, FunctionalProperty):
                domain = [Symbol]
                range = [bool]

            # --- BasicSymbol-specific properties ---
            class hasLevel(DataProperty, FunctionalProperty):
                domain = [BasicSymbol]
                range = [str]

            class hasDirection(DataProperty, FunctionalProperty):
                domain = [BasicSymbol]
                range = [str]

            # --- EffortSymbol-specific properties ---
            class hasWeightEffort(DataProperty, FunctionalProperty):
                domain = [EffortSymbol]
                range = [str]

            class hasTimeEffort(DataProperty, FunctionalProperty):
                domain = [EffortSymbol]
                range = [str]

            class hasSpaceEffort(DataProperty, FunctionalProperty):
                domain = [EffortSymbol]
                range = [str]

            class hasFlowEffort(DataProperty, FunctionalProperty):
                domain = [EffortSymbol]
                range = [str]

            # --- Annotation Property for Actions ---
            class label(AnnotationProperty, FunctionalProperty): pass

        # Define practical SWRL reasoning rules
        define_reasoning_rules(onto)
        define_action_classification_rules(onto)

        # Save ontology after creation
        onto.save(file=path)

    return onto

# Define reasoning rules using SWRL
def define_reasoning_rules(onto):
    with onto:
        
        # Symbols are simultaneous if intervals intersect
        rule0=Imp()
        rule0.set_as_rule("""
            Symbol(?sym1), Symbol(?sym2), differentFrom(?sym1, ?sym2),
            hasStartTime(?sym1, ?start1), hasEndTime(?sym1, ?end1),
            hasStartTime(?sym2, ?start2), hasEndTime(?sym2, ?end2),
            lessThanOrEqual(?start1, ?end2),
            lessThanOrEqual(?start2, ?end1)
            -> isSimultaneousWith(?sym1, ?sym2)
        """)

        # # Define previous and next symbol relationships per joint type
        rule1=Imp()
        rule1.set_as_rule("""
             Symbol(?sym1), Symbol(?sym2), differentFrom(?sym1, ?sym2),
             hasJointType(?sym1, ?joint), hasJointType(?sym2, ?joint),
             hasEndTime(?sym1, ?end1), hasStartTime(?sym2, ?start2),
             equal(?end1, ?start2)
             -> previousSymbol(?sym2, ?sym1), nextSymbol(?sym1, ?sym2)
         """)
    
        # # Identify start and end symbols
        rule2=Imp()
        rule2.set_as_rule("""
            Staff(?s), hasStartTime(?s, ?staffStart),
            hasSymbol(?s, ?sym), hasStartTime(?sym, ?t),
            equal(?t, ?staffStart)
            -> isStartSymbol(?sym, true)
        """)

        rule3=Imp()
        rule3.set_as_rule("""
            Staff(?s), hasEndTime(?s, ?staffEnd),
            hasSymbol(?s, ?sym), hasStartTime(?sym, ?t),
            equal(?t, ?staffStart)
            -> isEndSymbol(?sym, true)
        """)


        # Calculate staff interval based on start and end symbols
        # rule4=Imp()
        # rule4.set_as_rule("""
        #     Staff(?staff), hasSymbol(?staff, ?startSym), hasSymbol(?staff, ?endSym),
        #     isStartSymbol(?startSym, true), isEndSymbol(?endSym, true),
        #     hasStartTime(?startSym, ?staffStart), hasEndTime(?endSym, ?staffEnd)
        #     -> hasStartTime(?staff, ?staffStart), hasEndTime(?staff, ?staffEnd)
        # """)

        # Deduce action interval from defining symbols
        
        rule5=Imp()
        rule5.set_as_rule("""
            Action(?a), hasSymbol(?a, ?sym),
            hasStartTime(?sym, ?symStart), hasEndTime(?sym, ?symEnd)
            -> hasStartTime(?a, ?symStart), hasEndTime(?a, ?symEnd)
        """)



def define_action_classification_rules(onto):
    with onto:
        # Define the Action subclass 'Jump'
        class Jump(onto.Action): pass

        # Define the SWRL rule for identifying a 'Jump'
        jump_rule = Imp()
        jump_rule.set_as_rule("""
        BasicSymbol(?sym1), BasicSymbol(?sym2),
        hasLevel(?sym1, "Air"), hasLevel(?sym2, "Air"),
        hasJointType(?sym1, "LeftLeg"), hasJointType(?sym2, "RightLeg"),
        isSimultaneousWith(?sym1, ?sym2),
        hasStartTime(?sym1, ?s1), hasEndTime(?sym1, ?e1),
        hasStartTime(?sym2, ?s2), hasEndTime(?sym2, ?e2)
        -> Jump(?a), hasSymbol(?a, ?sym1), hasSymbol(?a, ?sym2),
        hasStartTime(?a, ?s1), hasEndTime(?a, ?e1)
        """)
        # Append the rule to the ontology
        # onto.rules.append(jump_rule)

# Query symbols associated with a specific joint, ordered by start time
def query_column(onto, joint_name):
    symbols = list(onto.search(hasJointType=joint_name))
    return sorted(symbols, key=lambda sym: sym.hasStartTime[0])

# Main execution example
if __name__ == '__main__':
    onto_path="jump_laban.owl"
    # Load or create the ontology
    onto = load_or_create_ontology(onto_path)


    # Run the reasoner to infer new knowledge based on defined rules
    sync_reasoner()

    # Example query usage
    left_leg_symbols = query_column(onto, "LeftLeg")
    for sym in left_leg_symbols:
        print(f"Symbol: {sym}, Start Time: {sym.hasStartTime[0]}, End Time: {sym.hasEndTime[0]}")
