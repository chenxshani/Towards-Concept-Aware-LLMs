"""
This script uses mturk's API to create a new HIT with tailor made test question as qualification
"""

import boto3


sandbox_host = 'mechanicalturk.sandbox.amazonaws.com'
region_name = 'us-east-1'
aws_access_key_id = "AKIAI6NQSPMAYYGETH2A"  # "AKIAI6NQSPMAYYGETH2A" # sela: AKIA4AIZD46RKHAC3MMX
aws_secret_access_key = "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"   # "De6+ULmjJJK/E2ytbAPLTzH3sr/n9QnbqP0UCTp7" # sela: ArXpy88Uou3SWpJOv/0dlRyZQgKyxNHnqK3p6C5J"
# endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com' # sandbox
# Uncomment this line to use in production
endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'


#################### Qualifications ####################
def   def_questions():
    """
    Test questions
    """
    questions = """
    <QuestionForm xmlns='http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2017-11-06/QuestionForm.xsd'>
            
        <Question>
          <QuestionIdentifier>Query1</QuestionIdentifier>
          <DisplayName>I can't get home for the holidays because of the _______.</DisplayName>
          <IsRequired>true</IsRequired>
          <QuestionContent>    

        <Text>
        You will be presented with a sentence containing a missing word and a concept/category.
        Your role is to determine whether completing the sentence with a word belonging to this concept/category would be: 
        likely / possible but not likely / does not make sense at all. 
        Meaning, given a sentence and a concept-word, we want you to: 
        Think about the word --> think about similar word, belonging to the same concept/category --> ask yourself whether one of them would make a good sentence-completion.
        
        Important: We want you to say whether a word *from this category* would make a good completion, not the specific word we present you with.
        Note! If the concept is not garammaticaly correct ("I enjoy *raining*" instead of *rain*) that is fine, we do not care about grammar here. 
        But if the sentence + a word from this concept is not a full sentence ("I enjoy *doing*") that is NOT fine, as the sentence is meaningless.
        
        Example 1
        Sentence: I went to the parent teacher conference with my _____.
        Concept/category of the completion: niece
        Desired response: Likely
        Explanation: Niece belongs to the general concept of *family*, which is very likely in this context
        
        Example 2
        Sentence: I went to the parent teacher conference with my _____.
        Concept/category of the completion: schedule
        Desired response: Does not make sense 
        Explanation: Schedule as a concept refers to *time*, meetings, etc. Nothing that makes sense in this context
        
        Example 3
        Sentence: I bought a fake _____ from a street vendor.
        Concept/category of the completion: music
        Desired response: Possible but not likely 
        Explanation: Music can represent a CD / music player, which is something people can but usually don't purchase from a street vendor + it can be fake
        
        Example 4
        Sentence: I bought a fake _____ from a street vendor.
        Concept/category of the completion: baby
        Desired response: Does not make sense 
        Explanation: Baby refers to the general concept of *people*. It's impossible to buy people on the street
                   
 --------------------------------------------------------------------------------------------------           
            Sentence: I can't get home for the holidays because of the _______.
            Concept/category of the completion: temperature
            </Text>

                  </QuestionContent>
                  <AnswerSpecification>
                    <SelectionAnswer>
                      <StyleSuggestion>radiobutton</StyleSuggestion>
                      <Selections>
                        <Selection>
                          <SelectionIdentifier>A1</SelectionIdentifier>
                          <Text>Likely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>A2</SelectionIdentifier>
                          <Text>Possible but unlikely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>A3</SelectionIdentifier>
                          <Text>Does not make sense</Text>
                        </Selection>
                      </Selections>
                    </SelectionAnswer>
                  </AnswerSpecification>
    </Question>


    <Question>
        <QuestionIdentifier>Query2</QuestionIdentifier>
        <DisplayName>I bought my ______ on credit.</DisplayName>
    <IsRequired>true</IsRequired>
    <QuestionContent>
        <Text>
        Sentence: I bought my ______ on credit.
        Concept/category of the completion: machine
        </Text>
                  </QuestionContent>
                  <AnswerSpecification>
                    <SelectionAnswer>
                      <StyleSuggestion>radiobutton</StyleSuggestion>
                      <Selections>
                        <Selection>
                          <SelectionIdentifier>B1</SelectionIdentifier>
                          <Text>Likely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>B2</SelectionIdentifier>
                          <Text>Possible but unlikely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>B3</SelectionIdentifier>
                          <Text>Does not make sense</Text>
                        </Selection>
                      </Selections>
                    </SelectionAnswer>
                  </AnswerSpecification>
        </Question>


    <Question>
        <QuestionIdentifier>Query3</QuestionIdentifier>
        <DisplayName>I need to prepare a speech for the ______.</DisplayName>
    <IsRequired>true</IsRequired>
    <QuestionContent>
        <Text>
        Sentence: I need to prepare a speech for the ______.
        Concept/category of the completion: appear
        </Text>
                  </QuestionContent>
                  <AnswerSpecification>
                    <SelectionAnswer>
                      <StyleSuggestion>radiobutton</StyleSuggestion>
                      <Selections>
                        <Selection>
                          <SelectionIdentifier>C1</SelectionIdentifier>
                          <Text>Likely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>C2</SelectionIdentifier>
                          <Text>Possible but unlikely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>C3</SelectionIdentifier>
                          <Text>Does not make sense</Text>
                        </Selection>
                      </Selections>
                    </SelectionAnswer>
                  </AnswerSpecification>
          </Question>    


    <Question>
        <QuestionIdentifier>Query4</QuestionIdentifier>
        <DisplayName>I squeezed myself into the ______.</DisplayName>
    <IsRequired>true</IsRequired>
    <QuestionContent>
        <Text>
        Sentence: I squeezed myself into the ______.
        Concept/category of the completion: sand
        </Text>
                  </QuestionContent>
                  <AnswerSpecification>
                    <SelectionAnswer>
                      <StyleSuggestion>radiobutton</StyleSuggestion>
                      <Selections>
                        <Selection>
                          <SelectionIdentifier>D1</SelectionIdentifier>
                          <Text>Likely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>D2</SelectionIdentifier>
                          <Text>Possible but unlikely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>D3</SelectionIdentifier>
                          <Text>Does not make sense</Text>
                        </Selection>
                      </Selections>
                    </SelectionAnswer>
                  </AnswerSpecification>
          </Question>    


    <Question>
        <QuestionIdentifier>Query5</QuestionIdentifier>
        <DisplayName>When I get back home after running outside in the sun I usually ______.</DisplayName>
    <IsRequired>true</IsRequired>
    <QuestionContent>
        <Text>
        Sentence: When I get back home after running outside in the sun I usually ______.
        Concept/category of the completion: water
        </Text>
                  </QuestionContent>
                  <AnswerSpecification>
                    <SelectionAnswer>
                      <StyleSuggestion>radiobutton</StyleSuggestion>
                      <Selections>
                        <Selection>
                          <SelectionIdentifier>E1</SelectionIdentifier>
                          <Text>Likely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>E2</SelectionIdentifier>
                          <Text>Possible but unlikely</Text>
                        </Selection>
                        <Selection>
                          <SelectionIdentifier>E3</SelectionIdentifier>
                          <Text>Does not make sense</Text>
                        </Selection>
                      </Selections>
                    </SelectionAnswer>
                  </AnswerSpecification>
          </Question>   
    </QuestionForm>
    """
    return questions


def def_answers():
    """
    Answers to the test questions (ground-truth scores). Make sure to update QualificationValueMapping according to the max score.
    """
    answers = """
    <AnswerKey xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/AnswerKey.xsd">
      <Question>
        <QuestionIdentifier>Query1</QuestionIdentifier>
        <AnswerOption>
          <SelectionIdentifier>A1</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
        <AnswerOption>
          <SelectionIdentifier>A2</SelectionIdentifier>
          <AnswerScore>0</AnswerScore>
        </AnswerOption>
            <AnswerOption>
          <SelectionIdentifier>A3</SelectionIdentifier>
          <AnswerScore>0</AnswerScore>
        </AnswerOption>
      </Question>


      <Question>
        <QuestionIdentifier>Query2</QuestionIdentifier>
        <AnswerOption>
          <SelectionIdentifier>B1</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
        <AnswerOption>
          <SelectionIdentifier>B2</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
            <AnswerOption>
          <SelectionIdentifier>B3</SelectionIdentifier>
          <AnswerScore>0</AnswerScore>
        </AnswerOption>
      </Question>


      <Question>
        <QuestionIdentifier>Query3</QuestionIdentifier>
        <AnswerOption>
          <SelectionIdentifier>C1</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
        <AnswerOption>
          <SelectionIdentifier>C2</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
            <AnswerOption>
          <SelectionIdentifier>C3</SelectionIdentifier>
          <AnswerScore>0</AnswerScore>
        </AnswerOption>
      </Question>


      <Question>
        <QuestionIdentifier>Query4</QuestionIdentifier>
        <AnswerOption>
          <SelectionIdentifier>D1</SelectionIdentifier>
          <AnswerScore>0</AnswerScore>
        </AnswerOption>
        <AnswerOption>
          <SelectionIdentifier>D2</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
            <AnswerOption>
          <SelectionIdentifier>D3</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
      </Question>


      <Question>
        <QuestionIdentifier>Query5</QuestionIdentifier>
        <AnswerOption>
          <SelectionIdentifier>E1</SelectionIdentifier>
          <AnswerScore>1</AnswerScore>
        </AnswerOption>
        <AnswerOption>
          <SelectionIdentifier>E2</SelectionIdentifier>
          <AnswerScore>0</AnswerScore>
        </AnswerOption>
            <AnswerOption>
          <SelectionIdentifier>E3</SelectionIdentifier>
          <AnswerScore>0</AnswerScore>
        </AnswerOption>
      </Question>


      <QualificationValueMapping>
        <PercentageMapping>
          <MaximumSummedScore>5</MaximumSummedScore> 
        </PercentageMapping>
      </QualificationValueMapping>
    </AnswerKey>
    """
    return answers


def create_qualifications(client, q_name, questions, answers, existing_qualification=False):
    """
    Creates the qualifications for the HIT we'll create.
    "localRequirements" defined all qualifications where "qual_type_ID" is our tailored made test questions qualification.
    """
    if existing_qualification:
        with open("test_qualification_id_facts.txt", "r") as id_file:
            qual_type_ID = id_file.readlines()[0]
            print("Using existed test qualification with ID:", qual_type_ID)
    else:
        qual_response = client.create_qualification_type(
            Name=q_name,
            Keywords='test, classification',
            Description="You will be presented with a sentence containing a missing word and a concept/category."
                        "Your role is to determine whether completing the sentence with a word belonging to this concept/category would be:"
                        "[likely, possible but unlikely, doesn't make sense]."
                        'Note! If the concept is not garammaticaly correct ("I enjoy *raining*" instead of *rain*) that is fine, we do not care about grammar here.'
                        'But if the sentence + a word from this concept is not a full sentence ("I enjoy *doing*") that is NOT fine, as the sentence is meaningless.You have a sentence with a missing word and a possible word to fill-in the blank.',
            QualificationTypeStatus='Active',
            Test=questions,
            AnswerKey=answers,
            TestDurationInSeconds=600)

        qual_type_ID = qual_response['QualificationType']['QualificationTypeId']
        print(qual_type_ID)

    localRequirements = [
        {'QualificationTypeId': '00000000000000000071',  #location
         'Comparator': 'In',
         'LocaleValues': [{'Country': 'US'}]},
        {'QualificationTypeId': '000000000000000000L0',  #% approved
         'Comparator': 'GreaterThanOrEqualTo',
         'IntegerValues': [97]},
        {'QualificationTypeId': '00000000000000000040',  # num approved
         'Comparator': 'GreaterThan',
         'IntegerValues': [2000]},
        {'QualificationTypeId': qual_type_ID,
         'Comparator': 'GreaterThan',
         'IntegerValues': [99]}  # allowing 1/6 mistakes
        # ,'RequiredToPreview': False
        # }
    ]
    return localRequirements


def main():
    # Establishing mturk connection
    client = boto3.client('mturk',
                          endpoint_url=endpoint_url,
                          region_name=region_name,
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key)

    print(client.get_account_balance())  # [$10,000.00]
    q_name = "Would words related to this concept/category make reasonable sentence completions?"
    questions = def_questions()
    answers = def_answers()

    localRequirements = create_qualifications(client, q_name, questions, answers)


if __name__ == "__main__":
    main()


