package water.rapids.ast.prims.mungers;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.Vec;
import water.rapids.Env;
import water.rapids.Merge;
import water.rapids.ast.AstPrimitive;
import water.rapids.ast.AstRoot;
import water.rapids.ast.params.AstNum;
import water.rapids.ast.params.AstNumList;
import water.rapids.vals.ValFrame;
import water.util.IcedHashMap;

import java.util.Arrays;


/** Given a dataframe, a list of groupby columns, a list of sort columns, a list of sort directions, a string
 * for the new name of the rank column, this class
 * will sort the whole dataframe according to the columns and sort directions.  It will add the rank of the
 * row within the groupby groups based on the sorted order determined by the sort columns and sort directions
 */
public class AstRankWithinGroupBy extends AstPrimitive {
  @Override public String[] args() {
    return new String[]{"frame", "groupby_cols", "sort_cols", "sort_orders", "new_colname"};
  }

  @Override public String str(){ return "rank_within_groupby";}
  @Override public int nargs() { return 1+5; } // (rank_within_groupby frame groupby_cols sort_cols sort_orders new_colname)

  @Override public ValFrame apply(Env env, Env.StackHelp stk, AstRoot asts[]) {
    Frame fr = stk.track(asts[1].exec(env)).getFrame(); // first argument is dataframe
    int[] groupbycols = ((AstNumList) asts[2]).expand4();  // groupby columns
    int[] sortcols =((AstNumList) asts[3]).expand4();  // sort columns
    int[] sortAsc;
    if (asts[4] instanceof AstNumList)
      sortAsc = ((AstNumList) asts[4]).expand4();
    else
      sortAsc = new int[]{(int) ((AstNum) asts[4]).getNum()};  // R client can send 1 element for some reason
    String newcolname = asts[5].exec(env).getStr();
    
    assert sortAsc.length==sortcols.length;
    SortnGrouby sortgroupbyrank = new SortnGrouby(fr, groupbycols, sortcols, sortAsc, newcolname).doAll(fr);  // sort and add rank column
  // finish ranking here
    return new ValFrame(Merge.sort(fr,sortcols, sortAsc));
  }

  public class SortnGrouby extends MRTask<SortnGrouby> {
    final int[] _sortCols;
    final int[] _groupbyCols;
    final int[] _sortOrders;
    final String _newColname;
    Frame _groupedSortedOut;  // store final result
    IcedHashMap<double[], Integer>[] _chunkStas;  // store all groupby class per chunk
    final int _groupbyLen;

    private SortnGrouby(Frame original, int[] groupbycols, int[] sortCols, int[] sortasc, String newcolname) {
      _sortCols = sortCols;
      _groupbyCols = groupbycols;
      _sortOrders = sortasc;
      _newColname = newcolname;
      _groupedSortedOut = Merge.sort(_groupedSortedOut,_sortCols, _sortOrders); // sort frame
      Vec newrank = original.anyVec().makeCon(0);
      _groupedSortedOut.add(_newColname, newrank);  // add new rank column of zeros
      int numChunks = _groupedSortedOut.vec(0).nChunks();
      _chunkStas = new IcedHashMap[numChunks];  // one HashMap per chunk to count groups and number per group
      _groupbyLen = _groupbyCols.length;
    }


    @Override
    public void map(Chunk[] chunks) {
      int cidx = chunks[0].cidx();  // grab chunk id
      int chunkLen = chunks[0].len();
      _chunkStas[cidx] = new IcedHashMap<>(); // create new HashMap for this chunk
      double[] keys = new double[_groupbyCols.length];  // allocate key memory
      for (int rind=0; rind<chunkLen; rind++) { // go through each row and groups them
        for (int colInd = 0; colInd < _groupbyLen; colInd++) {
          keys[colInd] = chunks[_groupbyCols[colInd]].atd(rind);
        }
        if (_chunkStas[cidx].containsKey(keys)) { // found this before
          int oldVal = _chunkStas[cidx].get(keys);
          _chunkStas[cidx].replace(keys, oldVal, oldVal+1);
        } else {  // new value
          _chunkStas[cidx].putIfAbsent(Arrays.copyOf(keys, _groupbyLen), 1);
        }
      }
    }

    @Override
    public void reduce(SortnGrouby git){  // copy over the information from one chunk to the final
      int numChunks = git._chunkStas.length;  // total number of chunks existed
      for (int ind = 0; ind < git._chunkStas.length; ind++) {
        if (git._chunkStas[ind] != null && git._chunkStas[ind].size() > 0) {
          _chunkStas[ind] = new IcedHashMap<>();  // copy over to current chunk stats
          for (double[] keys:_chunkStas[ind].keySet()) {
            _chunkStas[ind].put(Arrays.copyOf(keys,_groupbyLen),  git._chunkStas[ind].get(keys));
          }
        }
      }
    }

  }
}
