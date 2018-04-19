import sys, os
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from random import randint

def glm_multinomial_mojo_pojo():
    train,groupCols,sortCols = generate_trainingFrame()
    answerFrame = generate_answerFrame(train, groupCols, sortCols) # the rank_within_group result should return this





    glmMultinomialModel = pyunit_utils.build_save_model_GLM(params, x, train, "response") # build and save mojo model

    MOJONAME = pyunit_utils.getMojoName(glmMultinomialModel._id)
    TMPDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath('__file__')), "..", "results", MOJONAME))

    h2o.download_csv(test[x], os.path.join(TMPDIR, 'in.csv'))  # save test file, h2o predict/mojo use same file
    pred_h2o, pred_mojo = pyunit_utils.mojo_predict(glmMultinomialModel, TMPDIR, MOJONAME)  # load model and perform predict
    h2o.download_csv(pred_h2o, os.path.join(TMPDIR, "h2oPred.csv"))
    pred_pojo = pyunit_utils.pojo_predict(glmMultinomialModel, TMPDIR, MOJONAME)
    print("Comparing mojo predict and h2o predict...")
    pyunit_utils.compare_frames_local(pred_h2o, pred_mojo, 0.1, tol=1e-10)    # make sure operation sequence is preserved from Tomk        h2o.save_model(glmOrdinalModel, path=TMPDIR, force=True)  # save model for debugging
    print("Comparing pojo predict and h2o predict...")
    pyunit_utils.compare_frames_local(pred_mojo, pred_pojo, 0.1, tol=1e-10)


def generate_trainingFrame():
    train = pyunit_utils.random_dataset("regression", verbose=False, NTESTROWS=0)
    trainGroup = pyunit_utils.random_dataset_enums_only(train.nrows, 1, randint(2,20))
    trainEnums = pyunit_utils.random_dataset_enums_only(train.nrows, randint(1,3), randint(50,1000))   # columns to sort
    sortColumnsNames = ["sort0", "sort1", "sort2"]
    trainEnums.set_names(sortColumnsNames[0:trainEnums.ncols])
    trainGroup.set_name(0, "GroupByCols")
    finalTrain = train.cbind(trainGroup).cbind(trainEnums) # this will be the training frame
    return finalTrain,trainGroup.names,trainEnums.names

def generate_answerFrame(originalFrame, groupByCols, sortCols):
    """
    Given a dataset, a list of groupBy column names or indices and a list of sort column names or indices, this
    function will return a dataframe that is sorted according to the columns in sortCols and a new column is added
    to the frame that indicates the rank of the row within the groupBy columns sorted according to the sortCols.

    :param originalFrame:
    :param groupByCols: 
    :param sortCols:
    :return:
    """
    answerFrame = originalFrame.sort(sortCols)
    temp1 = answerFrame.as_data_frame(use_pandas=False) # change to data_frame for speedup
    rankwithGroup = dict()  # key value pair to keep track of rank within groupby groups
    groupLen = len(groupByCols)
    nrows = answerFrame.nrow
    tempKeys = range(groupLen)
    finalRank = range(originalFrame.nrow)
    for row in range(1,nrows):
        for col in range(groupLen):
            tempKeys[col]=temp1[row][col]
        keyDict = tuple(tempKeys)
        if (rankwithGroup.has_key(keyDict)):
            rankwithGroup[keyDict] += 1
        else:
            rankwithGroup[keyDict] = 1
        finalRank[row-1] = rankwithGroup[keyDict]
    rankFrame = h2o.H2OFrame(python_obj=finalRank)
    rankFrame.set_name(0,"new_rank_within_group")
    answerFrame = answerFrame.cbind(rankFrame)
    return answerFrame


if __name__ == "__main__":
    pyunit_utils.standalone_test(glm_multinomial_mojo_pojo)
else:
    glm_multinomial_mojo_pojo()
