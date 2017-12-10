/**
 * Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
 *
 */

package KKH.StdLib;

import KKH.StdLib.Interfaces_LamdaFunctions.functor_double_doubleArray;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.moment.*;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.DataBufferUShort;
import java.io.*;
import java.util.*;
import java.util.function.DoubleSupplier;

public final class Matk implements Serializable {

    private double[] data;
    private int nr;
    private int nc;
    private int nch;
    private int ndata_per_chan;
    private int ndata;

    // for continuous slicing
    private boolean isSubmat;
    private int r1;
    private int r2;
    private int c1;
    private int c2;
    private int ch1;
    private int ch2;

    // return type of methods min() and max()
    public static class Result_minmax
    {
        public double val;
        public int i, j, k;
    }

    // return type of methods kmeans_pp(...)
    public static class Result_clustering
    {
        public Matk centroids; // centroids, one column is for one centroid
        public Matk labels; // cluster labels for each data point
        public int nclusters; // final number of clusters after clustering
    }

    // return types of methods sort(...)
    public static class Result_sort
    {
        public Matk matSorted;
        public Matk indices_sort;
    }

    // return types of methods min(String...)
    public static class Result_minMax_eachDim
    {
        public Matk matVals;
        public Matk matIndices;
    }

    // return types of methods
    public static class Result_labelled_data
    {
        public Matk dataset;
        public Matk labels;
    }

    // return type of methods find(...)
    public static class Result_find
    {
        // linear indices assuming col major order where giving condition was found
        public int[] indices;
        // i component of ijk locations
        public int[] iPos;
        // j component of ijk locations
        public int[] jPos;
        // k component of ijk locations
        public int[] kPos;
        // the values in this matrix which satisfied the find condition
        public double[] vals;
        // number of found elements
        public int nFound;
    }

    private void ini_matrixHeader()
    {
        isSubmat = false;
        r1 = 0;
        r2 = nr - 1;
        c1 = 0;
        c2 = nc - 1;
        ch1 = 0;
        ch2 = nch - 1;
    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(double[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            System.arraycopy( data, 0, this.data, 0, ndata );
        }
        else
        {
            for(int k=0; k<nch; k++)
            {
                int cc = 0;
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = data[i * nch * nc + j * nch + k];
            }
        }

        ini_matrixHeader();
    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(float[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            for(int ii=0; ii<ndata; ii++)
                this.data[ii] = (double)data[ii];
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (double)data[i * nch * nc + j * nch + k];
        }

        ini_matrixHeader();
    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(int[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            for(int ii=0; ii<ndata; ii++)
                this.data[ii] = (double)data[ii];
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (double)data[i * nch * nc + j * nch + k];
        }

        ini_matrixHeader();
    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * Will treat the byte as "unsigned char" which will be represented as 0-255 in the
     * internal double array.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(byte[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            for(int ii=0; ii<ndata; ii++)
                this.data[ii] = (double)(data[ii] & 0xFF);
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (double)(data[i * nch * nc + j * nch + k] & 0xFF);
        }

        ini_matrixHeader();
    }

    // makes a column vector
    public Matk(double[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        System.arraycopy( data, 0, this.data, 0, ndata );

        ini_matrixHeader();
    }

    // makes a column vector
    public Matk(float[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;
        for(int ii=0; ii<ndata; ii++)
            temp[ii] = (double)data[ii];

        ini_matrixHeader();
    }

    // makes a column vector
    public Matk(int[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;
        for(int ii=0; ii<ndata; ii++)
            temp[ii] = (double)data[ii];

        ini_matrixHeader();
    }

    // makes a column vector
    public Matk(byte[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;
        for(int ii=0; ii<ndata; ii++)
            temp[ii] = (double)(data[ii] & 0xFF);

        ini_matrixHeader();
    }

    /**
     * Create a matrix of zeros of given dimensions
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(int nrows, int ncols, int nchannels, boolean allocate_data)
    {
        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        if(allocate_data)
            data = new double[ndata];

        ini_matrixHeader();
    }

    public Matk(int nrows, int ncols, int nchannels)
    {
        this(nrows, ncols, nchannels, true);
    }

    public Matk(int nrows, int ncols)
    {
        this(nrows, ncols, 1);
    }

    public Matk(int nrows)
    {
        this(nrows, 1, 1);
    }

    public Matk(double[][] m)
    {
        nr = m.length;
        nc = m[0].length;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = m[i][j];

        ini_matrixHeader();
    }

    public Matk(double[][][] m)
    {
        nch = m.length;
        nr = m[0].length;
        nc = m[0][0].length;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = m[k][i][j];

        ini_matrixHeader();
    }

    public Matk(float[][] m)
    {
        nr = m.length;
        nc = m[0].length;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = (double)m[i][j];

        ini_matrixHeader();
    }

    public Matk(float[][][] m)
    {
        nch = m.length;
        nr = m[0].length;
        nc = m[0][0].length;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = (double)m[k][i][j];

        ini_matrixHeader();
    }

    public Matk(int[][] m)
    {
        nr = m.length;
        nc = m[0].length;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = (double)m[i][j];

        ini_matrixHeader();
    }

    public Matk(int[][][] m)
    {
        nch = m.length;
        nr = m[0].length;
        nc = m[0][0].length;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = (double)m[k][i][j];

        ini_matrixHeader();
    }

    public <T extends Number> Matk(List<T> data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.size() != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = data.get(k * nr * nc + j * nr + i).doubleValue();
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = data.get(i * nch * nc + j * nch + k).doubleValue();
        }

        ini_matrixHeader();
    }

    // makes a column_vector
    public <T extends Number> Matk(List<T> data)
    {
        nr = data.size();
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;

        for(int ii=0; ii<ndata; ii++)
            temp[ii] = data.get(ii).doubleValue();

        ini_matrixHeader();
    }

    public <T extends Number> Matk(T[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (Double)data[k * nr * nc + j * nr + i];
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (Double)data[i * nch * nc + j * nch + k];
        }

        ini_matrixHeader();
    }

    // makes a column_vector
    public <T extends Number> Matk(T[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;

        for(int ii=0; ii<ndata; ii++)
            temp[ii] = (Double)data[ii];

        ini_matrixHeader();
    }

    // construct from Apache Common Math RealMatrix
    public Matk(RealMatrix m)
    {
        nch = 1;
        nr = m.getRowDimension();
        nc = m.getColumnDimension();
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = m.getEntry(i,j);

        ini_matrixHeader();
    }

    // construct from an array of Apache Common Math RealMatrix
    // assume that all the matrices have the same size
    public Matk(RealMatrix[] mArr)
    {
        nch = mArr.length;
        nr = mArr[0].getRowDimension();
        nc = mArr[0].getColumnDimension();
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            RealMatrix m = mArr[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = m.getEntry(i,j);
        }

        ini_matrixHeader();
    }

    // construct from a list of Apache Common Math RealMatrix
    // assume that all the matrices have the same size
    // dummy can be anything; just to distinguish from another method
    public Matk(List<RealMatrix> mL, boolean dummy)
    {
        nch = mL.size();
        nr = mL.get(0).getRowDimension();
        nc = mL.get(0).getColumnDimension();
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            RealMatrix m = mL.get(k);
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = m.getEntry(i,j);
        }

        ini_matrixHeader();
    }

    public Matk(BufferedImage image)
    {
        nr = image.getHeight();
        nc = image.getWidth();
        int img_type = image.getType();
        int cc;

        switch ( img_type )
        {
            case BufferedImage.TYPE_BYTE_GRAY:
            case BufferedImage.TYPE_3BYTE_BGR:
            case BufferedImage.TYPE_4BYTE_ABGR:

                if(img_type == BufferedImage.TYPE_BYTE_GRAY) nch = 1;
                else if(img_type == BufferedImage.TYPE_3BYTE_BGR) nch = 3;
                else if(img_type == BufferedImage.TYPE_4BYTE_ABGR) nch = 4;
                final byte[] bb = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
                ndata_per_chan = nr * nc;
                ndata = nr * nc * nch;
                data = new double[ndata];
                cc = 0;
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                            data[cc++] = (double)(bb[i * nch * nc + j * nch + k] & 0xFF);
                break;

            case BufferedImage.TYPE_USHORT_GRAY:

                nch = 1;
                final short[] sb = ((DataBufferUShort)image.getRaster().getDataBuffer()).getData() ;
                ndata_per_chan = nr * nc;
                ndata = nr * nc * nch;
                data = new double[ndata];
                cc = 0;
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                            data[cc++] = (double)sb[i * nch * nc + j * nch + k];
                break;

            case BufferedImage.TYPE_INT_RGB:
            case BufferedImage.TYPE_INT_BGR:
            case BufferedImage.TYPE_INT_ARGB:

                nch = 3;
                if(img_type == BufferedImage.TYPE_INT_ARGB) nch = 4;
                final int[] ib = ((DataBufferInt)image.getRaster().getDataBuffer()).getData();
                ndata_per_chan = nr * nc;
                ndata = nr * nc * nch;
                data = new double[ndata];
                cc = 0;
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                            data[cc++] = (double)ib[i * nch * nc + j * nch + k];
                break;
        }

        ini_matrixHeader();
    }

    /**
     * Construct/load a Matkc from a file. If is_image is true, then treat the file
     * as an image and read it accordingly. Else, treat it as a file that has
     * been saved through serialization and load it accordingly.
     * @param file_path
     * @param is_image
     */
    public static Matk load(String file_path, boolean is_image)
    {
        if(is_image)
            return new Matk(imread(file_path));
        else
        {
            Matk e = null;
            try {
                FileInputStream fileIn = new FileInputStream(file_path);
                ObjectInputStream in = new ObjectInputStream(fileIn);
                e = (Matk) in.readObject();
                in.close();
                fileIn.close();
                return e;
            }catch(IOException i) {
                i.printStackTrace();
                throw new IllegalArgumentException("file_path cannot be read");
            }catch(ClassNotFoundException c) {
                c.printStackTrace();
                throw new IllegalArgumentException("file_path cannot be read");
            }
        }
    }

    public void save(String filepath)
    {
        try {
            FileOutputStream fileOut =
                    new FileOutputStream(filepath);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(this);
            out.close();
            fileOut.close();
            //System.out.println("Serialized data is saved in " + filepath);
        }catch(IOException i) {
            i.printStackTrace();
        }
    }

    /**
     * Convert this matrix to BufferedImage. Makes a copy of underlying data
     * @return BufferedImage of type of either BufferedImage.TYPE_BYTE_GRAY
     * BufferedImage.TYPE_3BYTE_BGR depending on whether this matrix
     * is 1 or 3 channels respectively.
     */
    public BufferedImage to_BufferedImage_type_byte()
    {
        int img_type;
        switch(nchannels())
        {
            case 1:
                img_type = BufferedImage.TYPE_BYTE_GRAY;
                break;
            case 3:
                img_type = BufferedImage.TYPE_3BYTE_BGR;
                break;
            default:
                throw new IllegalArgumentException("This matrix is not 1 or 3 channels. Cannot show as image.");
        }

        BufferedImage img = new BufferedImage(ncols(), nrows(), img_type);
        final byte[] targetPixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        int cc = 0;
        for(int i=0; i<nrows(); i++)
            for(int j=0; j<ncols(); j++)
                for(int k=0; k<nchannels(); k++)
                    targetPixels[cc++] = (byte)get(i,j,k);

        return img;
    }

    /**
     * Convert this matrix to BufferedImage
     * @param type_int Must be either BufferedImage.TYPE_INT_BGR or BufferedImage.TYPE_INT_RGB
     * @return BufferedImage of type either BufferedImage.TYPE_INT_BGR or BufferedImage.TYPE_INT_RGB
     * depending on type_int input.
     */
    public BufferedImage to_BufferedImage_type_int(int type_int)
    {
        if(nchannels() != 3)
            throw new IllegalArgumentException("ERROR: This matrix must have 3 channels.");

        if(type_int != BufferedImage.TYPE_INT_BGR || type_int != BufferedImage.TYPE_INT_RGB)
            throw new IllegalArgumentException("ERROR: type_int must be either BufferedImage.TYPE_INT_BGR or BufferedImage.TYPE_INT_RGB.");

        BufferedImage img = new BufferedImage(ncols(), nrows(), type_int);
        final int[] targetPixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
        int cc = 0;
        for(int i=0; i<nrows(); i++)
            for(int j=0; j<ncols(); j++)
                for(int k=0; k<nchannels(); k++)
                    targetPixels[cc++] = (int)get(i,j,k);

        return img;
    }

    public double[][][] to_double3DArray()
    {
        double[][][] mOut = new double[nchannels()][nrows()][ncols()];
        if(isSubmat)
        {
            for(int k=0; k<nchannels(); k++)
            {
                double[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = get(i,j,k);
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
            {
                double[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = data[cc++];
            }
        }

        return mOut;
    }

    public double[][] to_double2DArray()
    {
        if(nchannels() != 1)
            throw new IllegalArgumentException("ERROR: nchannels() != 1");
        double[][] mOut = new double[nrows()][ncols()];

        if(isSubmat)
        {
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = get(i,j,0);
        }
        else
        {
            int cc = 0;
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = data[cc++];
        }

        return mOut;
    }

    public double[] to_double1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * ncols() * nchannels();
        double[] vec = new double[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        vec[cc++] = get(i,j,k);
        }
        else
            System.arraycopy( data, 0, vec, 0, ndata );

        return vec;
    }

    public float[][][] to_float3DArray()
    {
        float[][][] mOut = new float[nchannels()][nrows()][ncols()];
        if(isSubmat)
        {
            for(int k=0; k<nchannels(); k++)
            {
                float[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (float)get(i,j,k);
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
            {
                float[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (float)data[cc++];
            }
        }

        return mOut;
    }

    public float[][] to_float2DArray()
    {
        if(nchannels() != 1)
            throw new IllegalArgumentException("ERROR: nchannels() != 1");
        float[][] mOut = new float[nrows()][ncols()];

        if(isSubmat)
        {
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = (float)get(i,j,0);
        }
        else
        {
            int cc = 0;
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = (float)data[cc++];
        }

        return mOut;
    }

    public float[] to_float1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * ncols() * nchannels();
        float[] vec = new float[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        vec[cc++] = (float)get(i,j,k);
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (float)data[ii];
        }

        return vec;
    }

    public int[][][] to_int3DArray()
    {
        int[][][] mOut = new int[nchannels()][nrows()][ncols()];
        if(isSubmat)
        {
            for(int k=0; k<nchannels(); k++)
            {
                int[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (int)get(i,j,k);
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
            {
                int[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (int)data[cc++];
            }
        }

        return mOut;
    }

    public int[][] to_int2DArray()
    {
        if(nchannels() != 1)
            throw new IllegalArgumentException("ERROR: nchannels() != 1");
        int[][] mOut = new int[nrows()][ncols()];

        if(isSubmat)
        {
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = (int)get(i,j,0);
        }
        else
        {
            int cc = 0;
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = (int)data[cc++];
        }

        return mOut;
    }

    public int[] to_int1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * ncols() * nchannels();
        int[] vec = new int[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        vec[cc++] = (int)get(i,j,k);
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (int)data[ii];
        }

        return vec;
    }

    public byte[][][] to_byte3DArray()
    {
        byte[][][] mOut = new byte[nchannels()][nrows()][ncols()];
        if(isSubmat)
        {
            for(int k=0; k<nchannels(); k++)
            {
                byte[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (byte)get(i,j,k);
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
            {
                byte[][] temp = mOut[k];
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (byte)data[cc++];
            }
        }

        return mOut;
    }

    public byte[][] to_byte2DArray()
    {
        if(nchannels() != 1)
            throw new IllegalArgumentException("ERROR: nchannels() != 1");
        byte[][] mOut = new byte[nrows()][ncols()];

        if(isSubmat)
        {
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = (byte)get(i,j,0);
        }
        else
        {
            int cc = 0;
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    mOut[i][j] = (byte)data[cc++];
        }

        return mOut;
    }

    public byte[] to_byte1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * ncols() * nchannels();
        byte[] vec = new byte[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for(int k=0; k<nchannels(); k++)
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                        vec[cc++] = (byte)get(i,j,k);
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (byte)data[ii];
        }

        return vec;
    }

    // convert to Apache Common Math RealMatrix
    public RealMatrix to_ACM_RealMatrix()
    {
        if(nchannels() != 1)
            throw new IllegalArgumentException("ERROR: nchannels() != 1");
        Array2DRowRealMatrix mOut = new Array2DRowRealMatrix(nrows(), ncols());
        double[][] m = mOut.getDataRef();
        if(isSubmat)
        {
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    m[i][j] = get(i,j,0);
        }
        else
        {
            int cc = 0;
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                    m[i][j] = data[cc++];
        }

        return mOut;
    }

    public static BufferedImage imread(String img_path)
    {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(img_path));
        } catch (IOException e) {
            System.out.println("ERROR: Could not read image file at img_path.");
        }
        return img;
    }

    public void imshow(String winTitle, int x_winLocation, int y_winLocation )
    {
        BufferedImage img = to_BufferedImage_type_byte();
        ImageIcon icon=new ImageIcon(img);
        JFrame frame=new JFrame(winTitle);
        JLabel lbl=new JLabel(icon);
        frame.add(lbl);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setLocation(x_winLocation, y_winLocation);
        frame.setVisible(true);
    }

    public void imshow(String winTitle)
    {
        imshow(winTitle, 0, 0);
    }

    public void imshow()
    {
        imshow("image", 0, 0);
    }

    /**
     * Create a matrix with uniformly distributed pseudorandom
     * integers between range [imin, imax].
     * similar to matlab's randi
     * @param nrows
     * @param ncols
     * @param nchannels
     * @param imin
     * @param imax
     * @return
     */
    public static Matk randi(int nrows, int ncols, int nchannels, int imin, int imax)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = (double)(rand.nextInt(imax + 1 - imin) + imin);
        return mOut;
    }

    /**
     * Uniformly distributed random numbers between continuous range rangeMin and rangeMax
     * similar to matlab's rand
     * @param nrows
     * @param ncols
     * @param nchannels
     * @param rangeMin
     * @param rangeMax
     * @return
     */
    public static Matk rand(int nrows, int ncols, int nchannels, double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Matk mOut = new Matk(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return mOut;
    }

    public static Matk rand(int nrows, int ncols, int nchannels)
    {
        return rand(nrows, ncols, nchannels, 0, 1);
    }

    /**
     * Normally distributed random numbers
     * similar to matlab's rand
     * @param nrows
     * @param ncols
     * @param nchannels
     * @param mean
     * @param std
     * @return
     */
    public static Matk randn(int nrows, int ncols, int nchannels, double mean, double std)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = rand.nextGaussian() * std + mean;
        return mOut;
    }

    public static Matk fill_ladder(int nrows, int ncols, int nchannels, double start_val, double step)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        for(int ii=0; ii<mOut.ndata(); ii++)
        {
            mOut.data[ii] = start_val;
            start_val += step;
        }
        return mOut;
    }

    public static Matk fill_ladder(int nrows, int ncols, int nchannels, double start_val)
    {
        return fill_ladder(nrows, ncols, nchannels, start_val, 1);
    }

    public static Matk fill_ladder(int nrows, int ncols, int nchannels)
    {
        return fill_ladder(nrows, ncols, nchannels, 0, 1);
    }

    /**
     * Gives same results as Matlab's linspace
     * @param start_val
     * @param end_val
     * @param nvals
     * @param vector_type
     * @return
     */
    public static Matk linspace(double start_val, double end_val, int nvals, String vector_type)
    {
        Matk mOut;

        switch(vector_type)
        {
            case "row":
                mOut = new Matk(1, nvals, 1);
                break;
            case "col":
                mOut = new Matk(nvals, 1, 1);
                break;
            case "channel":
                mOut = new Matk(1, 1, nvals);
                break;
            default:
                throw new IllegalArgumentException("ERROR: vector_type must be: \"row\", \"col\" or \"channel\"");
        }

        double step = (end_val - start_val) / (nvals - 1);

        double[] temp = mOut.data;
        for(int ii=0; ii<nvals; ii++)
        {
            temp[ii] = start_val;
            start_val += step;
        }

        return mOut;
    }

    public static Matk linspace(double start_val, double end_val, int nvals)
    {
        return linspace(start_val, end_val, nvals, "row");
    }

    public static Matk randn(int nrows, int ncols, int nchannels)
    {
        return randn(nrows, ncols, nchannels, 0, 1);
    }

    public static Matk ones(int nrows, int ncols, int nchannels, double val)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = val;
        return mOut;
    }

    public static Matk ones(int nrows, int ncols, int nchannels)
    {
        return ones(nrows, ncols, nchannels, 1);
    }

    public static Matk zeros(int nrows, int ncols, int nchannels)
    {
        return new Matk(nrows, ncols, nchannels);
    }

    /**
     * Make a deep copy of the current matrix. If the current matrix is a subview
     * the resulting matrix will not be a subview.
     * @return
     */
    public Matk copy_deep()
    {
        int nr_new = nrows();
        int nc_new = ncols();
        int nch_new = nchannels();
        int ndata_per_chan_new = nr_new * nc_new;

        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            if(nr_new == nr && nc_new == nc)
            {
                System.arraycopy( data, ch1 * ndata_per_chan, temp_out, 0, mOut.ndata );
            }
            else if(nr_new == nr && nc_new != nc)
            {
                int cc = 0;
                for(int k=0; k<nch_new; k++)
                {
                    System.arraycopy( data, (k + ch1) * ndata_per_chan + c1*nr, temp_out, cc, ndata_per_chan_new );
                    cc += nr_new * nc_new;
                }
            }
            else
            {
                int cc = 0;
                for(int k=0; k<nch_new; k++)
                    for(int j=0; j<nc_new; j++)
                    {
                        System.arraycopy( data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, temp_out, cc, nr_new );
                        cc += nr_new;
                    }
            }
        }
        else
            System.arraycopy( data, 0, mOut.data, 0, ndata );

        return mOut;
    }

    public int nrows()
    {
        return r2-r1+1;
    }

    public int ncols()
    {
        return c2-c1+1;
    }

    public int nchannels()
    {
        return ch2-ch1+1;
    }

    public int ndata() { return nrows() * ncols() * nchannels(); }

    public int ndata_per_chan() { return nrows() * ncols(); }

    public int length_vec()
    {
        if(!is_vector())
            throw new IllegalArgumentException("ERROR: this matrix is not a vector");

        return ndata();
    }

    public double[] data() { return data; }

    public boolean is_submat() { return isSubmat; }

    /**
     * Find out whether this matrix is a vector (either row, column or channel vector)
     * @return
     */
    public boolean is_vector()
    {
        // a vector is a 3D matrix for which two of the dimensions has length of one.
        int z1 = nrows() == 1 ? 1:0;
        int z2 = ncols() == 1 ? 1:0;
        int z3 = nchannels() == 1 ? 1:0;
        return z1+z2+z3 >= 2;
    }

    public boolean is_row_vector()
    {
        return nrows() == 1 && ncols() >= 1 && nchannels() == 1;
    }

    public boolean is_col_vector()
    {
        return nrows() >= 1 && ncols() == 1 && nchannels() == 1;
    }

    public boolean is_channel_vector()
    {
        return nrows() == 1 && ncols() == 1 && nchannels() >= 1;
    }

    // get the copy of data corresponding to given range of a full matrix (not view/submatrix)
    // and produces a full matrix
    // even though this is quite fast since using System.arraycopy
    // as much as possible
    public Matk extract_range(int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
        if(isSubmat)
            throw new IllegalArgumentException("ERROR: this matrix must NOT be a submatrix/view.");

        if(r1 == -1) r1 = nr-1;
        if(r2 == -1) r2 = nr-1;
        if(c1 == -1) c1 = nc-1;
        if(c2 == -1) c2 = nc-1;
        if(ch1 == -1) ch1 = nch-1;
        if(ch2 == -1) ch2 = nch-1;

        int nr_new = r2 - r1 + 1;
        int nc_new = c2 - c1 + 1;
        int nch_new = ch2 - ch1 + 1;
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        if(nr_new == nr && nc_new == nc)
        {
            System.arraycopy( data, ch1 * ndata_per_chan, temp_out, 0, ndata_new );
        }
        else if(nr_new == nr && nc_new != nc)
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
            {
                System.arraycopy( data, (k + ch1) * ndata_per_chan + c1*nr, temp_out, cc, ndata_per_chan_new );
                cc += nr_new * nc_new;
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
                for(int j=0; j<nc_new; j++)
                {
                    System.arraycopy( data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, temp_out, cc, nr_new );
                    cc += nr_new;
                }
        }

        return mOut;
    }

    public double get(int i, int j, int k)
    {
        return data[(k + ch1) * ndata_per_chan + (j + c1) * nr + (i + r1)];
    }

    // assume k=0
    public double get(int i, int j)
    {
        return data[ch1 * ndata_per_chan + (j + c1) * nr + (i + r1)];
    }

    // get from a linear index
    public double get(int lin_index)
    {
        if(isSubmat)
        {
            int num_rows = nrows();
            int num_cols = ncols();
            int num_data_per_chan = num_rows * num_cols;

            int k= (int)Math.floor((double)lin_index / (num_data_per_chan));
            int j = (int)Math.floor((double)(lin_index % (num_data_per_chan)) / num_rows);
            int i =(int)Math.floor(lin_index % num_rows);

            return data[(k + ch1) * ndata_per_chan + (j + c1) * nr + (i + r1)];
        }
        else
            return data[lin_index];
    }

    // assume j=0, k=0 for get
    public double get_from_col_vec(int i)
    {
        return data[ch1 * ndata_per_chan + c1 * nr + (i + r1)];
    }

    // assume i=0, k=0 for get
    public double get_from_row_vec(int j)
    {
        return data[ch1 * ndata_per_chan + (j + c1) * nr + r1];
    }

    public void set(double val, int i, int j, int k)
    {
        data[(k + ch1) * ndata_per_chan + (j + c1) * nr + (i + r1)] = val;
    }

    // assume k=0
    public void set(double val, int i, int j)
    {
        data[ch1 * ndata_per_chan + (j + c1) * nr + (i + r1)] = val;
    }

    // set data using a linear index assuming column major stored array
    public void set(double val, int lin_index)
    {
        if(isSubmat)
        {
            int num_rows = nrows();
            int num_cols = ncols();
            int num_data_per_chan = num_rows * num_cols;

            int k= (int)Math.floor((double)lin_index / (num_data_per_chan));
            int j = (int)Math.floor((double)(lin_index % (num_data_per_chan)) / num_rows);
            int i =(int)Math.floor(lin_index % num_rows);

            data[(k + ch1) * ndata_per_chan + (j + c1) * nr + (i + r1)] = val;
        }

        else
            data[lin_index] = val;
    }

    // assume j=0, k=0 for set
    public void set_in_col_vec(double val, int i)
    {
        data[ch1 * ndata_per_chan + c1 * nr + (i + r1)] = val;
    }

    // assume i=0, k=0 for set
    public void set_in_row_vec(double val, int j)
    {
        data[ch1 * ndata_per_chan + (j + c1) * nr + r1] = val;
    }

    /**
     * set the values of this matrix with another matrix
     * @param mIn the matrix whose values will be copied and set to this matrix.
     */
    public Matk set(Matk mIn)
    {
        if(mIn.nrows() != nrows() || mIn.ncols() != ncols()
                || mIn.nchannels() != nchannels())
            throw new IllegalArgumentException("ERROR: This matrix and the matrix to be set must have same sizes.");

        double[] temp_in = mIn.data;

        if(!isSubmat && !mIn.isSubmat)
        {
            System.arraycopy( temp_in, 0, data, 0, ndata );
        }

        else if(isSubmat && !mIn.isSubmat)
        {
            int cc = 0;

            for(int k=0; k<nchannels(); k++)
                for(int j=0; j<ncols(); j++)
                    for(int i=0; i<nrows(); i++)
                    {
                        //set(temp[cc++], i,j,k);
                        System.arraycopy(mIn.data, cc, data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, nrows());
                        cc += nrows();
                    }
        }

        else if(!isSubmat && mIn.isSubmat)
        {
//            int cc = 0;
//            double[] temp = mIn.data;
//            for(int k=0; k<nchannels(); k++)
//                for(int j=0; j<ncols(); j++)
//                    for(int i=0; i<nrows(); i++)
//                        data[cc++] =  mIn.get(i, j, k);

            int cc = 0;
            double[] temp = mIn.data;
            for(int k=0; k<nchannels(); k++)
                for(int j=0; j<ncols(); j++)
                    {
                        //data[cc++] =  mIn.get(i, j, k);
                        System.arraycopy(mIn.data, (k + mIn.ch1) * mIn.ndata_per_chan + (j + mIn.c1) * mIn.nr + mIn.r1, data, cc, nrows());
                        cc += nrows();
                    }
        }

        else
        {
            for(int k=0; k<nchannels(); k++)
                for(int j=0; j<ncols(); j++)
                    System.arraycopy(mIn.data, (k + mIn.ch1) * mIn.ndata_per_chan + (j + mIn.c1) * mIn.nr + mIn.r1, data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, nrows());

//            for(int k=0; k<nchannels(); k++)
//                for(int j=0; j<ncols(); j++)
//                    for(int i=0; i<nrows(); i++)
//                        set(mIn.get(i, j, k), i,j,k);

        }



//        int cc = 0;
//
//        double[] temp_in = mIn.data;
//        int ch1_in = mIn.ch1;
//        int ch2_in = mIn.ch2;
//        int c1_in = mIn.c1;
//        int c2_in = mIn.c2;
//        int r1_in = mIn.r1;
//        int r2_in = mIn.r2;
//        int ndata_per_chan_in = mIn.nr * mIn.nc;
//        int nr_in = mIn.nr;
//
//        if(!isSubmat && !mIn.isSubmat)
//        {
//            System.arraycopy( temp_in, 0, data, 0, ndata );
//        }
//
//        else if(!isSubmat && mIn.isSubmat)
//        {
//            for(int k=ch1_in; k<=ch2_in; k++)
//                for(int j=c1_in; j<=c2_in; j++)
//                    for(int i=r1_in; i<=r2_in; i++)
//                    {
//                        data[cc++] = temp_in[k * ndata_per_chan_in + j * nr_in + i];
//                    }
//
//        }
//
//        else if(isSubmat && !mIn.isSubmat)
//        {
//            for(int k=ch1; k<=ch2; k++)
//                for(int j=c1; j<=c2; j++)
//                    for(int i=r1; i<=r2; i++)
//                    {
//                        data[k * ndata_per_chan + j * nr + i] = temp_in[cc++];
//                    }
//        }
//
//        else
//        {
//            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
//                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
//                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
//                    {
//                        data[k * ndata_per_chan + j * nr + i] = temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
//                    }
//        }


        return this;
    }

    private double get_orig(int i, int j, int k)
    {
        return data[k * ndata_per_chan + j * nr + i];
    }

    private void set_orig(double val, int i, int j, int k)
    {
        data[k * ndata_per_chan + j * nr + i] = val;
    }

    public void print()
    {
        System.out.println("=========== Printing matrix ===========");
        for(int k=0; k<nchannels(); k++)
        {
            System.out.println("mat(:,:," + (k+1) + ")=[");
            for(int i=0; i<nrows(); i++)
            {
                for(int j=0; j<ncols()-1; j++)
                    System.out.print(get(i,j,k) + ",\t");
                System.out.println(get(i,ncols()-1,k) + ";");
            }
            System.out.println("];");
        }
        System.out.println("=========== Matrix printed ===========");
    }

    // take a continuous submatrix of either the original matrix
    // or the current submatrix
    // only change the matrix header. Data is just referenced.
    // Thus very fast operation
    public Matk submat(int r1_, int r2_, int c1_, int c2_, int ch1_, int ch2_)
    {
        Matk mOut = new Matk(nr, nc, nch);
        mOut.data = data;
        mOut.isSubmat = true;

        if(r1_ == -1) r1_ = nrows()-1;
        if(r2_ == -1) r2_ = nrows()-1;
        if(c1_ == -1) c1_ = ncols()-1;
        if(c2_ == -1) c2_ = ncols()-1;
        if(ch1_ == -1) ch1_ = nchannels()-1;
        if(ch2_ == -1) ch2_ = nchannels()-1;

        mOut.r1 = r1_ + r1;
        mOut.r2 = mOut.r1 + (r2_ - r1_);
        mOut.c1 = c1_ + c1;
        mOut.c2 = mOut.c1 + (c2_ - c1_);
        mOut.ch1 = ch1_ + ch1;
        mOut.ch2 = mOut.ch1 + (ch2_ - ch1_);

        return mOut;
    }

    public Matk submat(int r1_, int r2_, int c1_, int c2_)
    {
        return submat(r1_, r2_, c1_, c2_, 0, -1);
    }

    // take a discontinuous submatrix of the current matrix.
    // copy data to the new matrix with a fresh header.
    public Matk submat(int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;
        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[(channel_indices[k]+ch1) * ndata_per_chan + (col_indices[j]+c1) * nr + (row_indices[i]+r1)];

        return mOut;
    }

    // take a continuous submatrix in the form of a row
    public Matk row(int row_index)
    {
        return submat(row_index, row_index, 0, -1, 0, -1);
    }

    // take a continuous submatrix in the form of rows
    public Matk rows(int start_index, int end_index)
    {
        return submat(start_index, end_index, 0, -1, 0, -1);
    }

    // take a discontinuous submatrix in the form of rows
    public Matk rows(int[] row_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = c2 - c1 + 1;
        int nch_new = ch2 - ch1 + 1;
        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=ch1; k<=ch2; k++)
            for(int j=c1; j<=c2; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + j * nr + (row_indices[i] + r1)];

        return mOut;
    }

    // take a continuous submatrix in the form of a column
    public Matk col(int col_index)
    {
        return submat(0, -1, col_index, col_index, 0, -1);
    }

    // take a continuous submatrix in the form of cols
    public Matk cols(int start_index, int end_index)
    {
        return submat(0, -1, start_index, end_index, 0, -1);
    }

    // take a discontinuous submatrix in the form of cols
    public Matk cols(int[] col_indices)
    {
        int nr_new = r2 - r1 + 1;
        int nc_new = col_indices.length;
        int nch_new = ch2 - ch1 + 1;
        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=ch1; k<=ch2; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=r1; i<=r2; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + (col_indices[j] + c1) * nr + i];

        return mOut;

    }

    // take a continuous submatrix in the form of a channel
    public Matk channel(int channel_index)
    {
        return submat(0, -1, 0, -1, channel_index, channel_index);
    }

    // take a continuous submatrix in the form of channels
    public Matk channels(int start_index, int end_index)
    {
        return submat(0, -1, 0, -1, start_index, end_index);
    }

    // take a discontinuous submatrix in the form of cols
    public Matk channels(int[] channel_indices)
    {
        int nr_new = r2 - r1 + 1;
        int nc_new = c2 - c1 + 1;
        int nch_new = channel_indices.length;
        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=c1; j<=c2; j++)
                for(int i=r1; i<=r2; i++)
                    temp_out[cc++] = data[(channel_indices[k] + ch1) * ndata_per_chan + j * nr + i];

        return mOut;

    }

    /**
     * Flatten the current matrix to either a row, column or channel vector matrix
     * Always results in a copy.
     * @param target_vec Can be "row", "column" or "channel".
     *                   If row, will result in a row vector.
     *                   If column, will result in a col vector, etc.
     * @return
     */
    public Matk vectorize(String target_vec)
    {
        int nr_new = nrows();
        int nc_new = ncols();
        int nch_new = nchannels();
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        Matk mOut;

        switch(target_vec)
        {
            case "row":
                mOut = new Matk(1, ndata_new, 1);
                break;
            case "column":
                mOut = new Matk(ndata_new, 1, 1);
                break;
            case "channel":
                mOut = new Matk(1, 1, ndata_new);
                break;
            default:
                throw new IllegalArgumentException("ERROR: target_vec must be either \"row\", \"column\" or \"channel\".");
        }

        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            if(nr_new == nr && nc_new == nc)
            {
                System.arraycopy( data, ch1 * ndata_per_chan, temp_out, 0, mOut.ndata );
            }
            else if(nr_new == nr && nc_new != nc)
            {
                int cc = 0;
                for(int k=0; k<nch_new; k++)
                {
                    System.arraycopy( data, (k + ch1) * ndata_per_chan + c1*nr, temp_out, cc, ndata_per_chan_new );
                    cc += nr_new * nc_new;
                }
            }
            else
            {
                int cc = 0;
                for(int k=0; k<nch_new; k++)
                    for(int j=0; j<nc_new; j++)
                    {
                        System.arraycopy( data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, temp_out, cc, nr_new );
                        cc += nr_new;
                    }
            }
        }
        else
            System.arraycopy( data, 0, mOut.data, 0, ndata );


        return mOut;
    }

    /**
     * Similar as vectorize() but returns an array instead of Matkc
     * @return
     */
    public double[] vectorize_to_doubleArray()
    {
//        int ndata_new = nrows() * ncols() * nchannels();
//        double[] vec = new double[ndata_new];
//
//        if(isSubmat)
//        {
//            int cc = 0;
//            for(int k=0; k<nchannels(); k++)
//                for(int j=0; j<ncols(); j++)
//                    for(int i=0; i<nrows(); i++)
//                        vec[cc++] = get(i,j,k);
//        }
//        else
//        {
//            System.arraycopy( data, 0, vec, 0, ndata );
//        }

        int nr_new = nrows();
        int nc_new = ncols();
        int nch_new = nchannels();
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        double[] vec = new double[ndata_new];

        if(isSubmat)
        {
            if(nr_new == nr && nc_new == nc)
            {
                System.arraycopy( data, ch1 * ndata_per_chan, vec, 0, ndata_new );
            }
            else if(nr_new == nr && nc_new != nc)
            {
                int cc = 0;
                for(int k=0; k<nch_new; k++)
                {
                    System.arraycopy( data, (k + ch1) * ndata_per_chan + c1*nr, vec, cc, ndata_per_chan_new );
                    cc += nr_new * nc_new;
                }
            }
            else
            {
                int cc = 0;
                for(int k=0; k<nch_new; k++)
                    for(int j=0; j<nc_new; j++)
                    {
                        System.arraycopy( data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, vec, cc, nr_new );
                        cc += nr_new;
                    }
            }
        }
        else
            System.arraycopy( data, 0, vec, 0, ndata );

        return vec;
    }

    public float[] vectorize_to_floatArray()
    {
        int ndata_new = nrows() * ncols() * nchannels();
        float[] vec = new float[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for (int k = ch1; k <= ch2; k++)
                for (int j = c1; j <= c2; j++)
                    for (int i = r1; i <= r2; i++)
                        vec[cc++] = (float) data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (float)data[ii];
        }

        return vec;
    }

    public int[] vectorize_to_intArray()
    {
        int ndata_new = nrows() * ncols() * nchannels();
        int[] vec = new int[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for (int k = ch1; k <= ch2; k++)
                for (int j = c1; j <= c2; j++)
                    for (int i = r1; i <= r2; i++)
                        vec[cc++] = (int) data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (int)data[ii];
        }

        return vec;
    }

    public byte[] vectorize_to_byteArray()
    {
        int ndata_new = nrows() * ncols() * nchannels();
        byte[] vec = new byte[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for (int k = ch1; k <= ch2; k++)
                for (int j = c1; j <= c2; j++)
                    for (int i = r1; i <= r2; i++)
                        vec[cc++] = (byte) data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (byte)data[ii];
        }

        return vec;
    }

    public Double[] vectorize_to_DoubleArray()
    {
        int ndata_new = nrows() * ncols() * nchannels();
        Double[] vec = new Double[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for (int k = ch1; k <= ch2; k++)
                for (int j = c1; j <= c2; j++)
                    for (int i = r1; i <= r2; i++)
                        vec[cc++] = data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = data[ii];
        }

        return vec;
    }

    public Float[] vectorize_to_FloatArray()
    {
        int ndata_new = nrows() * ncols() * nchannels();
        Float[] vec = new Float[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for (int k = ch1; k <= ch2; k++)
                for (int j = c1; j <= c2; j++)
                    for (int i = r1; i <= r2; i++)
                        vec[cc++] = (float)data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (float)data[ii];
        }

        return vec;
    }

    public Integer[] vectorize_to_IntegerArray()
    {
        int ndata_new = nrows() * ncols() * nchannels();
        Integer[] vec = new Integer[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for (int k = ch1; k <= ch2; k++)
                for (int j = c1; j <= c2; j++)
                    for (int i = r1; i <= r2; i++)
                        vec[cc++] = (int)data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (int)data[ii];
        }

        return vec;
    }

    public Byte[] vectorize_to_ByteArray()
    {
        int ndata_new = nrows() * ncols() * nchannels();
        Byte[] vec = new Byte[ndata_new];

        if(isSubmat)
        {
            int cc = 0;
            for (int k = ch1; k <= ch2; k++)
                for (int j = c1; j <= c2; j++)
                    for (int i = r1; i <= r2; i++)
                        vec[cc++] = (byte)data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                vec[ii] = (byte)data[ii];
        }

        return vec;
    }

    /**
     * Transpose this matrix
     * @return tranposed matrix
     */
    public Matk t()
    {
        if(nchannels() != 1)
            throw new IllegalArgumentException("ERROR: Cannot transpose matrix with more than 1 channel");

        Matk mOut = new Matk(ncols(), nrows(), 1);
        double[] temp = mOut.data;

        int cc = 0;
        for(int i=r1; i<=r2; i++)
            for(int j=c1; j<=c2; j++)
                temp[cc++] = data[j * nr + i];

        return mOut;
    }

    public Matk increment_IP()
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] + 1;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = data[ii] + 1;
        }

        return this; // just for convenience
    }

    public Matk increment()
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                        temp_out[cc++] = data[k * ndata_per_chan + j * nr + i] + 1;
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = data[ii] + 1;
        }

        return mOut;
    }

    public Matk decrement_IP()
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] - 1;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = data[ii] - 1;
        }

        return this; // just for convenience
    }

    public Matk decrement()
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                        temp_out[cc++] = data[k * ndata_per_chan + j * nr + i] - 1;
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = data[ii] - 1;
        }

        return mOut;
    }

    /**
     * Test whether this object is equal to anther object.
     * Two matrices are considered equal if they have
     * same nrows, ncols, nchannels and exactly the same data values.
     * This will also work for views and non-views (full matrices).
     * @param obj_
     * @return
     */
    @Override
    public boolean equals(Object obj_) {

        // If the object is compared with itself then return true
        if (obj_ == this) {
            return true;
        }

        /* Check if o is an instance of Complex or not
          "null instanceof [type]" also returns false */
        if (!(obj_ instanceof Matk)) {
            return false;
        }

        // typecast o to Complex so that we can compare data members
        Matk mIn = (Matk) obj_;

        if( (mIn.nrows() != nrows() ) || (mIn.ncols() != ncols() ) ||
                (mIn.nchannels() != nchannels() ))
            return false;

        double val_1, val_2;

        int cc = 0;
        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
            {
                if(data[ii] != temp_in[ii]) return false;
            }
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        val_1 = data[cc++];
                        val_2 = temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        if(val_1 != val_2) return false;
                    }
        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        val_1 = data[k * ndata_per_chan + j * nr + i];
                        val_2 = temp_in[cc++];
                        if(val_1 != val_2) return false;
                    }
        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        val_1 = data[k * ndata_per_chan + j * nr + i];
                        val_2 = temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
                        if(val_1 != val_2) return false;
                    }
        }

        return true;
    }

    /**
     * Check whether two matrices are approximately similar up to
     * some given tolerance
     * @param mIn
     * @param tolerance given tolerance. E.g. 0.00001. The smaller
     *                  this number is, the more strict the comparison
     *                  becomes.
     * @return
     */
    public boolean equals_approx(Matk mIn, double tolerance) {

        // If the object is compared with itself then return true
        if (mIn == this) {
            return true;
        }

        if( (mIn.nrows() != nrows() ) || (mIn.ncols() != ncols() ) ||
                (mIn.nchannels() != nchannels() ))
            return false;


        double val_1, val_2;

        int cc = 0;
        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
            {
                if(Math.abs(data[ii] - temp_in[ii]) > tolerance) return false;
            }
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        val_1 = data[cc++];
                        val_2 = temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        if(Math.abs(val_1 - val_2) > tolerance) return false;
                    }
        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        val_1 = data[k * ndata_per_chan + j * nr + i];
                        val_2 = temp_in[cc++];
                        if(Math.abs(val_1 - val_2) > tolerance) return false;
                    }
        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        val_1 = data[k * ndata_per_chan + j * nr + i];
                        val_2 = temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
                        if(Math.abs(val_1 - val_2) > tolerance) return false;
                    }
        }


        return true;
    }

    /**
     * perform dot product between this matrix and given matrix.
     * Assume that the given matrices are column or row vectors.
     * @param mIn
     * @return
     */
    public double dot(Matk mIn)
    {
        if(ndata() != mIn.ndata())
            throw new IllegalArgumentException("ERROR: ndata() != mIn.ndata()");

        double sum = 0;

        int cc = 0;
        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            double temp[] = mIn.data;
            for(int ii=0; ii<ndata; ii++)
                sum += (data[ii] * temp[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
                for(int k=ch1_in; k<=ch2_in; k++)
                    for(int j=c1_in; j<=c2_in; j++)
                        for(int i=r1_in; i<=r2_in; i++)
                        {
                            sum += data[cc] * temp_in[k * ndata_per_chan_in + j * nr_in + i];
                            cc++;
                        }
        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        sum += data[k * ndata_per_chan + j * nr + i] * temp_in[cc];
                        cc++;
                    }
        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        sum += data[k * ndata_per_chan + j * nr + i] * temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
                    }
        }

        return sum;
    }

    /**
     * element-wise multiplication of two matrices
     * @param mIn
     * @return
     */
    public Matk multE(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;
        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = (data[ii] * temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        temp_out[cc] =  data[cc] * temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] * temp_in[cc];
                        cc++;
                    }

        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                        temp_out[cc++] = data[k * ndata_per_chan + j * nr + i] * temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
        }

        return mOut;
    }

    public Matk multE_IP(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = (data[ii] * temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        data[cc] =  data[cc] * temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] * temp_in[cc++];
                    }
        }

        else
        {
            int idx;
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] * temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
                    }

        }


        return this;
    }

    public Matk mult(double val)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] * val;
                        cc++;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = data[ii] * val;
        }

        return mOut;
    }

    public Matk mult_IP(double val)
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] * val;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = data[ii] * val;
        }


        return this;
    }

    /**
     * Multiply this matrix with another matrix.
     * @param mIn
     * @return
     */
    public Matk mult(Matk mIn)
    {
        if(nchannels() != 1 || mIn.nchannels() != 1)
            throw new IllegalArgumentException("ERROR: matrix multiplication can be performed on matrices with one channel.");

        if( ncols() != mIn.nrows() )
            throw new IllegalArgumentException("ERROR: Invalid sizes of matrices for mutiplication.");

        int nr_new = nrows();
        int nc_new = mIn.ncols();

        Matk mOut = new Matk(nr_new, nc_new, 1);
        double[] temp = mOut.data;
        int cc = 0;
        for(int j=0; j<nc_new; j++)
            for(int i=0; i<nr_new; i++)
                temp[cc++] = row(i).dot(mIn.col(j));

        return mOut;
    }

    public Matk divE(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;
        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = (data[ii] / temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        temp_out[cc] =  data[cc] / temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] / temp_in[cc];
                        cc++;
                    }

        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                        temp_out[cc++] = data[k * ndata_per_chan + j * nr + i] / temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
        }

        return mOut;
    }

    public Matk divE_IP(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = (data[ii] / temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        data[cc] =  data[cc] / temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] / temp_in[cc++];
                    }
        }

        else
        {
            int idx;
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] / temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
                    }

        }


        return this;
    }

    public Matk div(double val)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] / val;
                        cc++;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = data[ii] / val;
        }

        return mOut;
    }

    public Matk div_IP(double val)
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] / val;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = data[ii] / val;
        }


        return this;
    }

    public Matk pow(double val)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = Math.pow(data[k * ndata_per_chan + j * nr + i], val);
                        cc++;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = Math.pow(data[ii], val);
        }

        return mOut;
    }

    public Matk pow_IP(double val)
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.pow(data[idx], val);
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.pow(data[ii], val);
        }

        return this;
    }


    public Matk plus(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot add two matrices of different sizes.");

        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;
        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = (data[ii] + temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        temp_out[cc] =  data[cc] + temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] + temp_in[cc];
                        cc++;
                    }

        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                        temp_out[cc++] = data[k * ndata_per_chan + j * nr + i] + temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
        }

        return mOut;
    }

    public Matk plus_IP(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot add two matrices of different sizes.");

        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = (data[ii] + temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        data[cc] =  data[cc] + temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] + temp_in[cc++];
                    }
        }

        else
        {
            int idx;
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] + temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
                    }
        }

        return this;
    }

    public Matk plus(double val)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] + val;
                        cc++;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = data[ii] + val;
        }

        return mOut;
    }

    public Matk plus_IP(double val)
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] + val;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = data[ii] + val;
        }

        return this;
    }

    public Matk minus(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot subtract two matrices of different sizes.");

        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;
        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = (data[ii] - temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        temp_out[cc] =  data[cc] - temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] - temp_in[cc];
                        cc++;
                    }

        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                        temp_out[cc++] = data[k * ndata_per_chan + j * nr + i] - temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
        }

        return mOut;
    }

    public Matk minus_IP(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot subtract two matrices of different sizes.");

        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = (data[ii] - temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        data[cc] =  data[cc] - temp_in[k * ndata_per_chan_in + j * nr_in + i];
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] - temp_in[cc++];
                    }
        }

        else
        {
            int idx;
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] - temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in];
                    }
        }

        return this;
    }

    public Matk minus(double val)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = data[k * ndata_per_chan + j * nr + i] - val;
                        cc++;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = data[ii] - val;
        }

        return mOut;
    }

    public Matk minus_IP(double val)
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = data[idx] - val;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = data[ii] - val;
        }

        return this;
    }

    // Generate a matrix by replicating this matrix in a block-like fashion
    // similar to matlab's repmat
    public Matk repmat(int ncopies_row, int ncopies_col, int ncopies_ch)
    {
        int nrows_this = nrows();
        int ncols_this = ncols();
        int nchannels_this = nchannels();

        Matk matOut = new Matk(nrows_this*ncopies_row,
                ncols_this*ncopies_col, nchannels_this*ncopies_ch);

        int row1, row2, col1, col2, chan1, chan2;

        for (int k = 0; k < ncopies_ch; k++)
            for (int j = 0; j < ncopies_col; j++)
                for (int i = 0; i < ncopies_row; i++)
                {
                    row1 = i*nrows_this;
                    row2 = i*nrows_this + nrows_this - 1;
                    col1 = j*ncols_this;
                    col2 = j*ncols_this + ncols_this - 1;
                    chan1 = k*nchannels_this;
                    chan2 = k*nchannels_this + nchannels_this - 1;
                    matOut.submat(row1, row2, col1, col2, chan1, chan2).set(this);
                }

        return matOut;
    }

    /**
     * Reshape a matrix.
     * @param nrows_new
     * @param ncols_new
     * @param nchannels_new
     */
    public Matk reshape(int nrows_new, int ncols_new, int nchannels_new, boolean copy_data)
    {
        if (nrows_new * ncols_new * nchannels_new != ndata())
            throw new IllegalArgumentException("ERROR: nrows_new * ncols_new * nchannels_new != ndata().");

        if(!copy_data && isSubmat)
            throw new IllegalArgumentException("ERROR: This matrix is a submatrix/view and therefore cannot use reshape without copying data.");

        Matk mOut;

        if(copy_data)
        {
            mOut = new Matk(nrows_new, ncols_new, nchannels_new);
            double[] temp_out = mOut.data;

            int cc = 0;
            int nr_new = nrows();
            int nc_new = ncols();
            int nch_new = nchannels();
            int ndata_per_chan_new = nr_new * nc_new;

            if(isSubmat)
            {
                if(nr_new == nr && nc_new == nc)
                {
                    System.arraycopy( data, ch1 * ndata_per_chan, temp_out, 0, mOut.ndata );
                }
                else if(nr_new == nr && nc_new != nc)
                {
                    for(int k=0; k<nch_new; k++)
                    {
                        System.arraycopy( data, (k + ch1) * ndata_per_chan + c1*nr, temp_out, cc, ndata_per_chan_new );
                        cc += nr_new * nc_new;
                    }
                }
                else
                {
                    for(int k=0; k<nch_new; k++)
                        for(int j=0; j<nc_new; j++)
                        {
                            System.arraycopy( data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, temp_out, cc, nr_new );
                            cc += nr_new;
                        }
                }
            }

            else
                System.arraycopy( data, 0, temp_out, 0, ndata );
        }

        else
        {
            mOut = new Matk(nrows_new, ncols_new, nchannels_new, false);
            mOut.data = data;
        }

        return mOut;
    }

    public Matk reshape(int nrows_new, int ncols_new, int nchannels_new)
    {
        return reshape(nrows_new, ncols_new, nchannels_new, true);
    }

    public Matk round()
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                        temp_out[cc++] = Math.round(data[k * ndata_per_chan + j * nr + i]);
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = Math.round(data[ii]);
        }


        return mOut;
    }

    public Matk round_IP()
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.round(data[idx]);
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.round(data[ii]);
        }

        return this;
    }

    public Matk zeros_IP()
    {
        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = 0;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = 0;
        }

        return this;
    }

    public Matk ones_IP()
    {
        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = 1;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = 1;
        }

        return this;
    }

    public Matk fill_IP(double val)
    {
        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = val;
                    }
        }
        else
        {
            Arrays.fill(data, val);
        }

        return this;
    }

    /**
     * Fill this matrix with uniformly distributed pseudorandom
     * integers between range [imin, imax]
     * similar to matlab's randi
     * @param imin
     * @param imax
     * @return
     */
    public Matk randi_IP(int imin, int imax)
    {
        Random rand = new Random();

        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = (double)(rand.nextInt(imax + 1 - imin) + imin);
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = (double)(rand.nextInt(imax + 1 - imin) + imin);
        }

        return this;
    }

    public Matk randi(int imin, int imax)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        Random rand = new Random();
        double[] temp_out = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp_out[ii] = (double)(rand.nextInt(imax + 1 - imin) + imin);
        return mOut;
    }

    /**
     * Uniformly distributed random numbers between continuous range rangeMin and rangeMax
     * similar to matlab's rand
     * @param rangeMin
     * @param rangeMax
     * @return
     */
    public Matk rand(double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        Random rand = new Random();
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return mOut;
    }

    public Matk rand()
    {
        return rand(0, 1);
    }

    public Matk rand_IP(double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Random rand = new Random();

        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        }


        return this;
    }

    public Matk rand_IP()
    {
        return rand_IP(0, 1);
    }

    /**
     * Normally distributed random numbers
     * similar to matlab's randn
     * @param mean
     * @param std
     * @return
     */
    public Matk randn(double mean, double std)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        Random rand = new Random();
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp[ii] = rand.nextGaussian() * std + mean;
        return mOut;
    }

    public Matk randn()
    {
        return randn(0, 1);
    }

    public Matk randn_IP(double mean, double std)
    {
        Random rand = new Random();

        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = rand.nextGaussian() * std + mean;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = rand.nextGaussian() * std + mean;
        }


        return this;
    }

    public Matk randn_IP()
    {
        return randn_IP(0, 1);
    }

    public Matk rand_custom(DoubleSupplier functor)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        Random rand = new Random();
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp[ii] = functor.getAsDouble();
        return mOut;
    }

    public Matk rand_custom_IP(DoubleSupplier functor)
    {
        Random rand = new Random();

        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = functor.getAsDouble();
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = functor.getAsDouble();
        }

        return this;
    }

    public Matk fill_ladder(double start_val, double step)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
        {
            temp[ii] = start_val;
            start_val += step;
        }
        return mOut;
    }

    public Matk fill_ladder_IP(double start_val, double step)
    {
        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        data[k * ndata_per_chan + j * nr + i] = start_val;
                        start_val += step;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
            {
                data[ii] = start_val;
                start_val += step;
            }
        }

        return this;
    }

    public Result_sort sort(boolean sort_col, boolean sort_ascend)
    {
        if(nchannels()!=1)
            throw new IllegalArgumentException("ERROR: for sorting, this matrix must have only one channel.");

        int number_rows = nrows();
        int number_cols = ncols();

        Result_sort res = new Result_sort();

        res.matSorted = new Matk(number_rows, number_cols, 1);
        res.indices_sort = new Matk(number_rows, number_cols, 1);

        if (sort_col)
        {
            Double[] vals = new Double[number_rows];
            for (int j = 0; j < number_cols; j++)
            {
                Integer[] indices = stdfuncs.fill_ladder_Integer(number_rows, 0, 1);
                for(int ii=0; ii < number_rows; ii++)
                    vals[ii] = get(ii,j,0);

                if(sort_ascend)
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] > (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }
                else // descend
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] < (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }

                for(int ii=0; ii < number_rows; ii++)
                {
                    res.indices_sort.set(indices[ii], ii,j,0);
                    res.matSorted.set(vals[indices[ii]], ii,j,0);
                }
            }
        }
        else
        {
            Double[] vals = new Double[number_cols];
            for (int i = 0; i < number_rows; i++)
            {
                Integer[] indices = stdfuncs.fill_ladder_Integer(number_cols, 0, 1);
                for(int jj=0; jj < number_cols; jj++)
                    vals[jj] = get(i,jj,0);

                if(sort_ascend)
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] > (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }
                else // descend
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] < (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }

                for(int jj=0; jj < number_cols; jj++)
                {
                    res.indices_sort.set(indices[jj], i,jj,0);
                    res.matSorted.set(vals[indices[jj]], i,jj,0);
                }
            }
        }

        return res;
    }

    public Result_sort sort(boolean sort_col)
    {
        return sort(sort_col, true);
    }

    public Matk max(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: This matrix and input matrix do not have same size.");

        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;
        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = Math.max(data[ii], temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        temp_out[cc] =  Math.max(data[cc], temp_in[k * ndata_per_chan_in + j * nr_in + i]);
                        cc++;
                    }
        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = Math.max(data[k * ndata_per_chan + j * nr + i], temp_in[cc]);
                        cc++;
                    }
        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                        temp_out[cc++] = Math.max(data[k * ndata_per_chan + j * nr + i], temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in]);
        }


        return mOut;
    }

    public Matk max_IP(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot element-wise divide two matrices of different sizes.");

        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.max(data[ii], temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        data[cc] =  Math.max(data[cc], temp_in[k * ndata_per_chan_in + j * nr_in + i]);
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.max(data[idx], temp_in[cc++]);
                    }
        }

        else
        {
            int idx;
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.max(data[idx], temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in]);
                    }
        }


        return this;
    }

    public Matk max(double val)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = Math.max(data[k * ndata_per_chan + j * nr + i], val);
                        cc++;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = Math.max(data[ii], val);
        }

        return mOut;
    }

    public Matk max_IP(double val)
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.max(data[idx], val);
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.max(data[ii], val);
        }

        return this;
    }

    public Result_minMax_eachDim max(String process_dim)
    {
        int number_rows = nrows();
        int number_cols = ncols();
        int number_chans = nchannels();

        switch(process_dim)
        {
            case "col":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(1, number_cols, number_chans);
                res.matIndices = new Matk(1, number_cols, number_chans);
                double maxCur, idxMaxCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        maxCur = get(0, j, k);
                            idxMaxCur = 0;
                            for (int i = 1; i < number_rows; i++)
                            {
                            val = get(i, j, k);
                            if (val > maxCur)
                            {
                                idxMaxCur = i;
                                maxCur = val;
                            }
                        }
                        res.matVals.set(maxCur,0, j, k);
                        res.matIndices.set(idxMaxCur,0, j, k);
                    }
                return res;
            }
            case "row":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, 1, number_chans);
                res.matIndices = new Matk(number_rows, 1, number_chans);
                double maxCur, idxMaxCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        maxCur = get(i, 0, k);
                        idxMaxCur = 0;
                        for (int j = 1; j < number_cols; j++)
                        {
                            val = get(i, j, k);
                            if (val > maxCur)
                            {
                                idxMaxCur = j;
                                maxCur = val;
                            }
                        }
                        res.matVals.set(maxCur, i, 0, k);
                        res.matIndices.set(idxMaxCur, i, 0, k);
                    }
                return res;
            }
            case "channel":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, number_cols, 1);
                res.matIndices = new Matk(number_rows, number_cols, 1);
                double maxCur, idxMaxCur, val;
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        maxCur = get(i, j, 0);
                        idxMaxCur = 0;
                        for(int k=1; k< number_chans; k++)
                        {
                            val = get(i, j, k);
                            if (val > maxCur)
                            {
                                idxMaxCur = k;
                                maxCur = val;
                            }
                        }
                        res.matVals.set(maxCur, i, j, 0);
                        res.matIndices.set(idxMaxCur, i, j, 0);
                    }
                return res;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    public Result_minmax max()
    {
        Result_minmax res = new Result_minmax();
        res.val = get(0,0,0);
        res.i = 0;
        res.j = 0;
        res.k = 0;
        double val_cur;

        for(int k=ch1; k<=ch2; k++)
            for(int j=c1; j<=c2; j++)
                for(int i=r1; i<=r2; i++)
                {
                    val_cur = data[k * ndata_per_chan + j * nr + i];
                    if(val_cur > res.val)
                    {
                        res.val = val_cur;
                        res.i = i-r1;
                        res.j = j-c1;
                        res.k = k-ch1;
                    }
                }

        return res;
    }

    public Matk min(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: This matrix and input matrix do not have same size.");

        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;
        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = Math.min(data[ii], temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        temp_out[cc] =  Math.min(data[cc], temp_in[k * ndata_per_chan_in + j * nr_in + i]);
                        cc++;
                    }
        }

        else if(isSubmat && !mIn.isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = Math.min(data[k * ndata_per_chan + j * nr + i], temp_in[cc]);
                        cc++;
                    }
        }

        else
        {
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                        temp_out[cc++] = Math.min(data[k * ndata_per_chan + j * nr + i], temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in]);
        }


        return mOut;
    }

    public Matk min_IP(Matk mIn)
    {
        if( (nrows()!=mIn.nrows()) || (ncols()!=mIn.ncols())
                || (nchannels()!=mIn.nchannels()))
            throw new IllegalArgumentException("ERROR: Cannot element-wise divide two matrices of different sizes.");

        int cc = 0;

        double[] temp_in = mIn.data;
        int ch1_in = mIn.ch1;
        int ch2_in = mIn.ch2;
        int c1_in = mIn.c1;
        int c2_in = mIn.c2;
        int r1_in = mIn.r1;
        int r2_in = mIn.r2;
        int ndata_per_chan_in = mIn.nr * mIn.nc;
        int nr_in = mIn.nr;

        if(!isSubmat && !mIn.isSubmat)
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.min(data[ii], temp_in[ii]);
        }

        else if(!isSubmat && mIn.isSubmat)
        {
            for(int k=ch1_in; k<=ch2_in; k++)
                for(int j=c1_in; j<=c2_in; j++)
                    for(int i=r1_in; i<=r2_in; i++)
                    {
                        data[cc] =  Math.min(data[cc], temp_in[k * ndata_per_chan_in + j * nr_in + i]);
                        cc++;
                    }

        }

        else if(isSubmat && !mIn.isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.min(data[idx], temp_in[cc++]);
                    }
        }

        else
        {
            int idx;
            for(int k_in=ch1_in, k=ch1; k_in<=ch2_in; k_in++, k++)
                for(int j_in=c1_in, j=c1; j_in<=c2_in; j_in++, j++)
                    for(int i_in=r1_in, i=r1; i_in<=r2_in; i_in++, i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.min(data[idx], temp_in[k_in * ndata_per_chan_in + j_in * nr_in + i_in]);
                    }
        }


        return this;
    }

    public Matk min(double val)
    {
        Matk mOut = new Matk(nrows(), ncols(), nchannels());
        double[] temp_out = mOut.data;

        if(isSubmat)
        {
            int cc = 0;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        temp_out[cc] = Math.min(data[k * ndata_per_chan + j * nr + i], val);
                        cc++;
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                temp_out[ii] = Math.min(data[ii], val);
        }

        return mOut;
    }

    public Matk min_IP(double val)
    {
        if(isSubmat)
        {
            int idx;
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                    {
                        idx = k * ndata_per_chan + j * nr + i;
                        data[idx] = Math.min(data[idx], val);
                    }
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.min(data[ii], val);
        }

        return this;
    }

    public Result_minMax_eachDim min(String process_dim)
    {
        int number_rows = nrows();
        int number_cols = ncols();
        int number_chans = nchannels();

        switch(process_dim)
        {
            case "col":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(1, number_cols, number_chans);
                res.matIndices = new Matk(1, number_cols, number_chans);
                double minCur, idxMinCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        minCur = get(0, j, k);
                        idxMinCur = 0;
                        for (int i = 1; i < number_rows; i++)
                        {
                            val = get(i, j, k);
                            if (val < minCur)
                            {
                                idxMinCur = i;
                                minCur = val;
                            }
                        }
                        res.matVals.set(minCur,0, j, k);
                        res.matIndices.set(idxMinCur,0, j, k);
                    }
                return res;
            }
            case "row":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, 1, number_chans);
                res.matIndices = new Matk(number_rows, 1, number_chans);
                double minCur, idxMinCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        minCur = get(i, 0, k);
                        idxMinCur = 0;
                        for (int j = 1; j < number_cols; j++)
                        {
                            val = get(i, j, k);
                            if (val < minCur)
                            {
                                idxMinCur = j;
                                minCur = val;
                            }
                        }
                        res.matVals.set(minCur, i, 0, k);
                        res.matIndices.set(idxMinCur, i, 0, k);
                    }
                return res;
            }
            case "channel":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, number_cols, 1);
                res.matIndices = new Matk(number_rows, number_cols, 1);
                double minCur, idxMinCur, val;
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        minCur = get(i, j, 0);
                        idxMinCur = 0;
                        for(int k=1; k< number_chans; k++)
                        {
                            val = get(i, j, k);
                            if (val < minCur)
                            {
                                idxMinCur = k;
                                minCur = val;
                            }
                        }
                        res.matVals.set(minCur, i, j, 0);
                        res.matIndices.set(idxMinCur, i, j, 0);
                    }
                return res;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    // compute the minimum value in the entire matrix and the corresponding index (location)
    public Result_minmax min()
    {
        Result_minmax res = new Result_minmax();
        res.val = get(0,0,0);
        res.i = 0;
        res.j = 0;
        res.k = 0;
        double val_cur;

        for(int k=ch1; k<=ch2; k++)
            for(int j=c1; j<=c2; j++)
                for(int i=r1; i<=r2; i++)
                {
                    val_cur = data[k * ndata_per_chan + j * nr + i];
                    if(val_cur < res.val)
                    {
                        res.val = val_cur;
                        res.i = i-r1;
                        res.j = j-c1;
                        res.k = k-ch1;
                    }
                }

        return res;
    }

    /**
     * Compute univariate moments such as mean, variance, std
     * @param moment_type a string identifying the type of moment that is to be
     *                    computed. can be "mean", "std", "var", "GeometricMean",
     *                    "Kurtosis", "SecondMoment", "SemiVariance", "Skewness"
     * @param process_dim can be "col", "row" or "channel". If "col", then the
     *                    statistics for each column is computed and the results
     *                    is saved. If "row", then statics for each row is computed, etc.
     * @param isBiasCorrected only applies for "std", "var" and "SemiVariance". The
     *                        default is true. If true, compute sample statistics. If false
     *                        then population statistics. E.g. for variance, if sample
     *                        statistics, then variance = sum((x_i - mean)^2) / (n - 1) is
     *                        used to compute the statistics whereas, if population statistics,
     *                        then variance = sum((x_i - mean)^2) / n is used.
     * @return
     */
    public Matk moment(String moment_type, String process_dim, boolean isBiasCorrected)
    {
        int number_rows = nrows();
        int number_cols = ncols();
        int number_chans = nchannels();

        UnivariateStatistic statsObj;

        switch(moment_type)
        {
            case "mean":
                statsObj = new Mean();
                break;
            case "std":
                statsObj = new StandardDeviation(isBiasCorrected);
                break;
            case "var":
                statsObj = new Variance(isBiasCorrected);
                break;
            case "GeometricMean":
                statsObj = new GeometricMean();
                break;
            case "Kurtosis":
                statsObj = new Kurtosis();
                break;
            case "SecondMoment":
                statsObj = new SecondMoment();
                break;
            case "SemiVariance":
                statsObj = new SemiVariance(isBiasCorrected);
                break;
            case "Skewness":
                statsObj = new Skewness();
                break;

            default:
                throw new IllegalArgumentException("ERROR: Invalid moment_type.");
        }

        switch(process_dim)
        {
            case "col":
            {
                Matk matOut = new Matk(1, number_cols, number_chans);
                double[] vals = new double[number_rows];
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        for (int i = 0; i < number_rows; i++)
                            vals[i] = get(i, j, k);
                        matOut.set(statsObj.evaluate(vals),0,j,k);
                    }
                return matOut;
            }
            case "row":
            {
                Matk matOut = new Matk(number_rows, 1, number_chans);
                double[] vals = new double[number_cols];
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for (int j = 0; j < number_cols; j++)
                            vals[j] = get(i, j, k);
                        matOut.set(statsObj.evaluate(vals), i,0,k);
                    }
                return matOut;
            }
            case "channel":
            {
                Matk matOut = new Matk(number_rows, number_cols, 1);
                double[] vals = new double[number_chans];
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for(int k=0; k< number_chans; k++)
                            vals[k] = get(i, j, k);
                        matOut.set(statsObj.evaluate(vals), i,j,0);
                    }
                return matOut;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    public Matk moment(String moment_type, String process_dim)
    {
        return moment(moment_type, process_dim, true);
    }

    /**
     * Summarize each row, column or channel with a single number
     * @param process_dim can be "col", "row" or "channel". If "col", then the
     *                    summary for each column is computed and the results
     *                    is saved. If "row", then statics for each row is computed, etc.
     * @param functor a class that implements the functor_double_doubleArray interface
     *                which has only one member function "double apply(double[] x)
     * @return
     */
    public Matk summarize(String process_dim, functor_double_doubleArray functor)
    {
        int number_rows = nrows();
        int number_cols = ncols();
        int number_chans = nchannels();

        switch(process_dim)
        {
            case "col":
            {
                Matk matOut = new Matk(1, number_cols, number_chans);
                double[] vals = new double[number_rows];
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        for (int i = 0; i < number_rows; i++)
                            vals[i] = get(i, j, k);
                        matOut.set(functor.apply(vals),0,j,k);
                    }
                return matOut;
            }
            case "row":
            {
                Matk matOut = new Matk(number_rows, 1, number_chans);
                double[] vals = new double[number_cols];
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for (int j = 0; j < number_cols; j++)
                            vals[j] = get(i, j, k);
                        matOut.set(functor.apply(vals), i,0,k);
                    }
                return matOut;
            }
            case "channel":
            {
                Matk matOut = new Matk(number_rows, number_cols, 1);
                double[] vals = new double[number_chans];
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for(int k=0; k< number_chans; k++)
                            vals[k] = get(i, j, k);
                        matOut.set(functor.apply(vals), i,j,0);
                    }
                return matOut;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    public Matk median(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.percentile(ee, 0.5));
    }

    public Matk percentile(String process_dim, double p)
    {
        return summarize(process_dim, ee->StatUtils.percentile(ee, p));
    }

    public Matk mode(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.mode(ee)[0]);
    }

    public Matk product(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.product(ee));
    }

    // sum all the elements in this matrix
    public double sum()
    {
        double total = 0;

        if(isSubmat)
        {
            for(int k=ch1; k<=ch2; k++)
                for(int j=c1; j<=c2; j++)
                    for(int i=r1; i<=r2; i++)
                        total += data[k * ndata_per_chan + j * nr + i];
        }
        else
        {
            for(int ii=0; ii<ndata; ii++)
                total += data[ii];
        }

        return total;
    }

    public Matk sum(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sum(ee));
    }

    // Returns the sum of the natural logs of the entries in the input array, or Double.NaN if the array is empty.
    public Matk sumLog(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sumLog(ee));
    }

    //Returns the sum of the squares of the entries in the input array, or Double.NaN if the array is empty.
    public Matk sumSq(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sumSq(ee));
    }

    //Returns the variance of the entries in the input array, or Double.NaN if the array is empty.
    // population version = true means  ( sum((x_i - mean)^2) / n )
    // population version = false means  ( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim, boolean population_version)
    {
        if(population_version)
            return summarize(process_dim, ee->StatUtils.populationVariance(ee));
        else
            return summarize(process_dim, ee->StatUtils.variance(ee));
    }

    //Returns the variance of the entries in the input array, or Double.NaN if the array is empty.
    // ( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.variance(ee));
    }

    //Returns the variance of the entries in the input array, using the precomputed mean value.
    // population version = true means  ( sum((x_i - mean)^2) / n )
    // population version = false means  ( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim, double mean, boolean population_version)
    {
        if(population_version)
            return summarize(process_dim, ee->StatUtils.populationVariance(ee, mean));
        else
            return summarize(process_dim, ee->StatUtils.variance(ee, mean));
    }

    //Returns the variance of the entries in the input array, using the precomputed mean value.
    //( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim, double mean)
    {
        return summarize(process_dim, ee->StatUtils.variance(ee, mean));
    }

    /**
     * compute histogram of this matrix; considers all the data in the whole matrix
     * gives same results as Matlab's histcounts/histc
     * @param edges
     * @return
     */
    public double[] hist(double[] edges)
    {
        if(!stdfuncs.is_sorted_ascend(edges))
            throw new IllegalArgumentException("ERROR: edges must be sorted in ascending in ascending order");

        int nedges = edges.length;
        int nbins = nedges - 1;
        double[] h = new double[nbins];

        int idx_edge_last = nedges - 1;
        double curVal;

        int idx_ub, idx_bin;

        for(int k=0; k<nchannels(); k++)
            for(int j=0; j<ncols(); j++)
                for(int i=0; i<nrows(); i++)
                {
                    curVal = get(i,j,k);
                    idx_ub = stdfuncs.bs_upper_bound(edges, curVal);

                    // handle boundary case (left most side)
                    if (idx_ub == 0)
                    {
                        // data less than e1 (the first edge), so don't count
                        if (curVal < edges[0])
                            continue;
                    }

                    // handle boundary case (right most side)
                    if (idx_ub == nedges)
                    {
                        // data greater than the last edge, so don't count
                        if (curVal > edges[idx_edge_last])
                            continue;
                        // need to decrement since due to being at exactly edge final
                        --idx_ub;
                    }

                    idx_bin = idx_ub - 1;
                    ++h[idx_bin];
                }

        return h;
    }

    // join this matrix with the given matrix matIn horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public Matk add_cols(Matk matIn)
    {
        int nrows_new = Math.max(nrows(), matIn.nrows());
        int ncols_new = ncols() + matIn.ncols();
        int nch_new = Math.max(nchannels(), matIn.nchannels());
        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        matOut.submat(0, nrows()-1, 0, ncols()-1, 0, nchannels()-1).set(this);
        matOut.submat(0, matIn.nrows()-1, ncols(), ncols_new-1, 0, matIn.nchannels()-1).set(matIn);
        return matOut;
    }

    // merge an array of matrices horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_cols(Matk[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat[kk].nrows());
            ncols_new += vmat[kk].ncols();
            nch_new = Math.max(nch_new, vmat[kk].nchannels());
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nc_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.submat(0, vmat[kk].nrows() - 1, nc_count, nc_count + vmat[kk].ncols() - 1, 0, vmat[kk].nchannels() - 1).set(vmat[kk]);
            nc_count += vmat[kk].ncols();
        }

        return matOut;
    }

    // merge a list of matrices horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_cols(List<Matk> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat.get(kk).nrows());
            ncols_new += vmat.get(kk).ncols();
            nch_new = Math.max(nch_new, vmat.get(kk).nchannels());
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nc_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.submat(0, vmat.get(kk).nrows() - 1, nc_count, nc_count + vmat.get(kk).ncols() - 1, 0, vmat.get(kk).nchannels() - 1).set(vmat.get(kk));
            nc_count += vmat.get(kk).ncols();
        }

        return matOut;
    }

    // join this matrix with the given matrix matIn vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public Matk add_rows(Matk matIn)
    {
        int nrows_new = nrows() + matIn.nrows();
        int ncols_new = Math.max(ncols(), matIn.ncols());
        int nch_new = Math.max(nchannels(), matIn.nchannels());
        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        matOut.submat(0, nrows() - 1, 0, ncols() - 1, 0, nchannels() - 1).set(this);
        matOut.submat(nrows(), nrows_new - 1, 0, matIn.ncols() - 1, 0, matIn.nchannels() - 1).set(matIn);
        return matOut;
    }

    // merge an array of matrices vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_rows(Matk[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new += vmat[kk].nrows();
            ncols_new = Math.max(ncols_new, vmat[kk].ncols());
            nch_new = Math.max(nch_new, vmat[kk].nchannels());
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nr_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.submat(nr_count, nr_count + vmat[kk].nrows() - 1, 0, vmat[kk].ncols() - 1, 0, vmat[kk].nchannels() - 1).set(vmat[kk]);
            nr_count += vmat[kk].nrows();
        }

        return matOut;
    }

    // merge a list of matrices vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_rows(List<Matk> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new += vmat.get(kk).nrows();
            ncols_new = Math.max(ncols_new, vmat.get(kk).ncols());
            nch_new = Math.max(nch_new, vmat.get(kk).nchannels());
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nr_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.submat(nr_count, nr_count + vmat.get(kk).nrows() - 1, 0, vmat.get(kk).ncols() - 1, 0, vmat.get(kk).nchannels() - 1).set(vmat.get(kk));
            nr_count += vmat.get(kk).nrows();
        }

        return matOut;
    }

    // add channels to the this matrix.
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public Matk add_channels(Matk matIn)
    {
        int nrows_new = Math.max(nrows(), matIn.nrows());
        int ncols_new = Math.max(ncols(), matIn.ncols());
        int nch_new = nchannels() + matIn.nchannels();
        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        matOut.submat(0, nrows() - 1, 0, ncols() - 1, 0, nchannels() - 1).set(this);
        matOut.submat(0, matIn.nrows() - 1, 0, matIn.ncols() - 1, nchannels(), nch_new - 1).set(matIn);
        return matOut;
    }

    // merge channels of an array of matrices
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public static Matk merge_channels(Matk[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat[kk].nrows());
            ncols_new = Math.max(ncols_new, vmat[kk].ncols());
            nch_new += vmat[kk].nchannels();
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nch_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.submat(0, vmat[kk].nrows() - 1, 0, vmat[kk].ncols() - 1, nch_count, nch_count + vmat[kk].nchannels() - 1).set(vmat[kk]);
            nch_count += vmat[kk].nchannels();
        }

        return matOut;
    }

    // merge channels of a list of matrices
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public static Matk merge_channels(List<Matk> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat.get(kk).nrows());
            ncols_new = Math.max(ncols_new, vmat.get(kk).ncols());
            nch_new += vmat.get(kk).nchannels();
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nch_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.submat(0, vmat.get(kk).nrows() - 1, 0, vmat.get(kk).ncols() - 1, nch_count, nch_count + vmat.get(kk).nchannels() - 1).set(vmat.get(kk));
            nch_count += vmat.get(kk).nchannels();
        }

        return matOut;
    }

    // remove cols from this matrix
    public Matk del_cols(int[] indices_remove)
    {
        // row indices that I want to keep (keep all; 1,2,...,nr)
        int[] row_idxs_keep = stdfuncs.fill_ladder_int(nrows(), 0, 1);

        // channel indices that I want to keep (keep all; 1,2,...,nch)
        int[] ch_idxs_keep = stdfuncs.fill_ladder_int(nchannels(), 0, 1);

        // col indices to keep
        int[] col_idxs_all = stdfuncs.fill_ladder_int(ncols(), 0, 1);
        int[] col_idxs_keep = stdfuncs.set_diff(col_idxs_all, indices_remove);

        Matk matOut = submat(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove rows from this matrix
    public Matk del_rows(int[] indices_remove)
    {
        // col indices that I want to keep (keep all; 1,2,...,nr)
        int[] col_idxs_keep = stdfuncs.fill_ladder_int(ncols(), 0, 1);

        // channel indices that I want to keep (keep all; 1,2,...,nch)
        int[] ch_idxs_keep = stdfuncs.fill_ladder_int(nchannels(), 0, 1);

        // row indices to keep
        int[] row_idxs_all = stdfuncs.fill_ladder_int(nrows(), 0, 1);
        int[] row_idxs_keep = stdfuncs.set_diff(row_idxs_all, indices_remove);

        Matk matOut = submat(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove channels from this matrix
    public Matk del_channels(int[] indices_remove)
    {
        // row indices that I want to keep (keep all; 1,2,...,nr)
        int[] row_idxs_keep = stdfuncs.fill_ladder_int(nrows(), 0, 1);

        // col indices that I want to keep (keep all; 1,2,...,nr)
        int[] col_idxs_keep = stdfuncs.fill_ladder_int(ncols(), 0, 1);

        // channel indices to keep
        int[] ch_idxs_all = stdfuncs.fill_ladder_int(nchannels(), 0, 1);
        int[] ch_idxs_keep = stdfuncs.set_diff(ch_idxs_all, indices_remove);

        Matk matOut = submat(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove submatrix from this matrix
    public Matk del_submat(int[] row_indices_remove, int[] col_indices_remove, int[] channel_indices_remove)
    {
        // row indices to keep
        int[] row_idxs_all = stdfuncs.fill_ladder_int(nrows(), 0, 1);
        int[] row_idxs_keep = stdfuncs.set_diff(row_idxs_all, row_indices_remove);

        // col indices to keep
        int[] col_idxs_all = stdfuncs.fill_ladder_int(ncols(), 0, 1);
        int[] col_idxs_keep = stdfuncs.set_diff(col_idxs_all, col_indices_remove);

        // channel indices to keep
        int[] ch_idxs_all = stdfuncs.fill_ladder_int(nchannels(), 0, 1);
        int[] ch_idxs_keep = stdfuncs.set_diff(ch_idxs_all, channel_indices_remove);

        Matk matOut = submat(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // find the locations of the elements in this matrix that satisfied
    // given number comparison condition
    public Result_find find(String comp_operator, double val)
    {
        int ini_capacity = Math.max(Math.max(nrows(), ncols()), nchannels());
        List<Integer> indices_list = new ArrayList<>(ini_capacity);
        List<Integer> i_list = new ArrayList<>(ini_capacity);
        List<Integer> j_list = new ArrayList<>(ini_capacity);
        List<Integer> k_list = new ArrayList<>(ini_capacity);
        List<Double> val_list = new ArrayList<>(ini_capacity);
        int cc = 0;
        double val_cur;

        switch(comp_operator)
        {
            case "=":
                for(int k=ch1; k<=ch2; k++)
                    for(int j=c1; j<=c2; j++)
                        for(int i=r1; i<=r2; i++)
                        {
                            val_cur = data[k * ndata_per_chan + j * nr + i];
                            if(val_cur == val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case ">=":
                for(int k=ch1; k<=ch2; k++)
                    for(int j=c1; j<=c2; j++)
                        for(int i=r1; i<=r2; i++)
                        {
                            val_cur = data[k * ndata_per_chan + j * nr + i];
                            if(val_cur >= val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case "<=":
                for(int k=ch1; k<=ch2; k++)
                    for(int j=c1; j<=c2; j++)
                        for(int i=r1; i<=r2; i++)
                        {
                            val_cur = data[k * ndata_per_chan + j * nr + i];
                            if(val_cur <= val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case ">":
                for(int k=ch1; k<=ch2; k++)
                    for(int j=c1; j<=c2; j++)
                        for(int i=r1; i<=r2; i++)
                        {
                            val_cur = data[k * ndata_per_chan + j * nr + i];
                            if(val_cur > val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case "<":
                for(int k=ch1; k<=ch2; k++)
                    for(int j=c1; j<=c2; j++)
                        for(int i=r1; i<=r2; i++)
                        {
                            val_cur = data[k * ndata_per_chan + j * nr + i];
                            if(val_cur < val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            default:
                throw new IllegalArgumentException("ERROR: comp_operator must be \"=\", \">=\", \"<=\", \">\", \"<\"");
        }

        Result_find res = new Result_find();
        res.indices = stdfuncs.list_to_intArray(indices_list);
        res.iPos = stdfuncs.list_to_intArray(i_list);
        res.jPos = stdfuncs.list_to_intArray(j_list);
        res.kPos = stdfuncs.list_to_intArray(k_list);
        res.vals = stdfuncs.list_to_doubleArray(val_list);
        res.nFound = indices_list.size();

        return res;
    }

    // assume that this matrix is each data point stored as a col vector in a 2D matrix
    public Result_clustering kmeans_pp(int nclusters, int nMaxIters)
    {
        if(nchannels()!=1)
            throw new IllegalArgumentException("ERROR: this matrix has more than one channel.");
        KMeansPlusPlusClusterer kmObj = new KMeansPlusPlusClusterer(nclusters, nMaxIters);
        List<vecPointForACMCluster> tdata = new ArrayList<vecPointForACMCluster>(ncols());
        for(int j=0; j<ncols(); j++)
            tdata.add(new vecPointForACMCluster(col(j)));
        List<CentroidCluster<vecPointForACMCluster>> res = kmObj.cluster(tdata);

        Result_clustering res_cluster = new Result_clustering();

        res_cluster.centroids = new Matk(nrows(), nclusters);
        res_cluster.labels = new Matk(1, ncols());
        res_cluster.nclusters = res.size();

        for(int j=0; j<res_cluster.nclusters; j++)
            res_cluster.centroids.col(j).set(new Matk(res.get(j).getCenter().getPoint(), true, nrows(), 1, 1));

        Matk dists = new Matk(1, res_cluster.nclusters);
        double dist;

        for(int j=0; j<ncols(); j++)
        {
            Matk cur_datapoint = col(j);
            // for this data point index j, compute euclidean distance to each of the
            // centroids and save it in dists
            for(int i=0; i<res_cluster.nclusters; i++)
            {
                dist = cur_datapoint.minus(res_cluster.centroids.col(i)).pow(2).sum();
                dists.set(dist,0,i,0);
            }

            // get the min distance
            Result_minmax res_min = dists.min();
            // record the cluster label for this data point index j
            res_cluster.labels.set(res_min.j + 1,0, j, 0);
        }

        return res_cluster;
    }

    /**
     * Generate a dataset that contains bivariate gaussian data for different number of classes
     * For each class, one multivariate gaussian distribution
     * @param nclasses
     * @param ndata_per_class Number of data for each class
     * @return an array of two matrices: (2xN) the generated data, (1xN) the labels
     * where N = ndata_per_class * nclasses
     */
    public static Result_labelled_data gen_dataset_BVN(int nclasses, int ndata_per_class)
    {
        double[] mean = new double[2];
        double[][] covariance = {{1,0}, {0, 1}};
        double[] sample;
        double rangeMin = -10;
        double rangeMax = 10;
        Random rand = new Random();

        Result_labelled_data res = new Result_labelled_data();
        res.dataset = new Matk(2, nclasses * ndata_per_class, 1);
        res.labels = new Matk(1, nclasses * ndata_per_class, 1);

        for(int i=0; i<nclasses; i++)
        {
            mean[0] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
            mean[1] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
            MultivariateNormalDistribution dist = new MultivariateNormalDistribution(mean, covariance);
            for(int j=0; j<ndata_per_class; j++)
            {
                sample = dist.sample();
                res.dataset.col((i*ndata_per_class)+j).set(new Matk(sample, true, 2, 1, 1));
                res.labels.set(i+1,0, (i*ndata_per_class)+j, 0);
            }
        }

        return res;
    }

}


class vecPointForACMCluster implements Clusterable
{
    double[] data;

    vecPointForACMCluster(Matk m)
    {
        data = m.vectorize_to_doubleArray();
    }

    @Override
    public double[] getPoint() {
        return data;
    }

}
