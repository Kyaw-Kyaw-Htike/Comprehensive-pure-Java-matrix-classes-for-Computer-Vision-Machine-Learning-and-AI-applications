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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleSupplier;

public final class Matkc implements Serializable {

    private double[] data;
    private int nr;
    private int nc;
    private int nch;
    private int ndata_per_chan;
    private int ndata;

    // return type of methods min() and max()
    public static class Result_minmax
    {
        public double val;
        public int i, j, k;
    }

    // return type of methods kmeans_pp(...)
    public static class Result_clustering
    {
        public Matkc centroids; // centroids, one column is for one centroid
        public Matkc labels; // cluster labels for each data point
        public int nclusters; // final number of clusters after clustering
    }

    // return types of methods sort(...)
    public static class Result_sort
    {
        public Matkc matSorted;
        public Matkc indices_sort;
    }

    // return types of methods min(String...)
    public static class Result_minMax_eachDim
    {
        public Matkc matVals;
        public Matkc matIndices;
    }

    // return types of methods
    public static class Result_labelled_data
    {
        public Matkc dataset;
        public Matkc labels;
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

    public Matkc()
    {
        nr = 0;
        nc = 0;
        nch = 0;
        ndata_per_chan = 0;
        ndata = 0;
        data = new double[ndata];
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
    public Matkc(double[] data, boolean col_major, int nrows, int ncols, int nchannels)
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
            double[] temp_in = this.data;
            for(int k=0; k<nch; k++)
            {
                int cc = 0;
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        temp_in[cc++] = data[i * nch * nc + j * nch + k];
            }
        }

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
    public Matkc(float[] data, boolean col_major, int nrows, int ncols, int nchannels)
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
    public Matkc(int[] data, boolean col_major, int nrows, int ncols, int nchannels)
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
    public Matkc(byte[] data, boolean col_major, int nrows, int ncols, int nchannels)
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
    }

    // makes a column vector
    public Matkc(double[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        System.arraycopy( data, 0, this.data, 0, ndata );

    }

    // makes a column vector
    public Matkc(float[] data)
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
    }

    // makes a column vector
    public Matkc(int[] data)
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
    }

    // makes a column vector
    public Matkc(byte[] data)
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
    }

    /**
     * Create a matrix of zeros of given dimensions
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matkc(int nrows, int ncols, int nchannels)
    {
        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        data = new double[ndata];
    }

    public Matkc(int nrows, int ncols)
    {
        this(nrows, ncols, 1);
    }

    public Matkc(int nrows)
    {
        this(nrows, 1, 1);
    }

    public Matkc(double[][] m)
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
    }

    public Matkc(double[][][] m)
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
    }

    public Matkc(float[][] m)
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
    }

    public Matkc(float[][][] m)
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
    }

    public Matkc(int[][] m)
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
    }

    public Matkc(int[][][] m)
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
    }

    public <T extends Number> Matkc(List<T> data, boolean col_major, int nrows, int ncols, int nchannels)
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

    }

    // makes a column_vector
    public <T extends Number> Matkc(List<T> data)
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
    }

    public <T extends Number> Matkc(T[] data, boolean col_major, int nrows, int ncols, int nchannels)
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
    }

    // makes a column_vector
    public <T extends Number> Matkc(T[] data)
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
    }

    // construct from Apache Common Math RealMatrix
    public Matkc(RealMatrix m)
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
    }

    // construct from an array of Apache Common Math RealMatrix
    // assume that all the matrices have the same size
    public Matkc(RealMatrix[] mArr)
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
    }

    // construct from a list of Apache Common Math RealMatrix
    // assume that all the matrices have the same size
    // dummy can be anything; just to distinguish from another method
    public Matkc(List<RealMatrix> mL, boolean dummy)
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

    }

    public Matkc(BufferedImage image)
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

    }

    /**
     * Construct/load a Matkc from a file. If is_image is true, then treat the file
     * as an image and read it accordingly. Else, treat it as a file that has
     * been saved through serialization and load it accordingly.
     * @param file_path
     * @param is_image
     */
    public static Matkc load(String file_path, boolean is_image)
    {
        if(is_image)
            return new Matkc(imread(file_path));
        else
        {
            Matkc e = null;
            try {
                FileInputStream fileIn = new FileInputStream(file_path);
                ObjectInputStream in = new ObjectInputStream(fileIn);
                e = (Matkc) in.readObject();
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

    // save col major data as a linear array in a text file
    public void save_data_txt(String fpath)
    {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fpath))) {

            bw.write(String.format("Matkc matrix: nrows = %d, ncols = %d, nchannels = %d\n", nr, nc, nch));
            for(int ii=0; ii<ndata; ii++)
                bw.write(data[ii] + "\n");

        } catch (IOException e) {
            e.printStackTrace();
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
        switch(nch)
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

        BufferedImage img = new BufferedImage(nc, nrows(), img_type);
        final byte[] targetPixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        int cc = 0;
        for(int i=0; i<nrows(); i++)
            for(int j=0; j<nc; j++)
                for(int k=0; k<nch; k++)
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
        if(nch != 3)
            throw new IllegalArgumentException("ERROR: This matrix must have 3 channels.");

        if(type_int != BufferedImage.TYPE_INT_BGR || type_int != BufferedImage.TYPE_INT_RGB)
            throw new IllegalArgumentException("ERROR: type_int must be either BufferedImage.TYPE_INT_BGR or BufferedImage.TYPE_INT_RGB.");

        BufferedImage img = new BufferedImage(nc, nrows(), type_int);
        final int[] targetPixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
        int cc = 0;
        for(int i=0; i<nrows(); i++)
            for(int j=0; j<nc; j++)
                for(int k=0; k<nch; k++)
                    targetPixels[cc++] = (int)get(i,j,k);

        return img;
    }

    public double[][][] to_double3DArray()
    {
        double[][][] mOut = new double[nch][nrows()][nc];

        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            double[][] temp = mOut[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nrows(); i++)
                    temp[i][j] = data[cc++];
        }

        return mOut;
    }

    public double[][] to_double2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        double[][] mOut = new double[nrows()][nc];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = data[cc++];

        return mOut;
    }

    public double[] to_double1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * nc * nch;
        double[] vec = new double[ndata_new];
        System.arraycopy( data, 0, vec, 0, ndata );
        return vec;
    }

    public float[][][] to_float3DArray()
    {
        float[][][] mOut = new float[nch][nrows()][nc];
            int cc = 0;
            for(int k=0; k<nch; k++)
            {
                float[][] temp = mOut[k];
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (float)data[cc++];
            }
        return mOut;
    }

    public float[][] to_float2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        float[][] mOut = new float[nrows()][nc];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = (float)data[cc++];

        return mOut;
    }

    public float[] to_float1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * nc * nch;
        float[] vec = new float[ndata_new];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (float)data[ii];
        return vec;
    }

    public int[][][] to_int3DArray()
    {
        int[][][] mOut = new int[nch][nrows()][nc];
        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            int[][] temp = mOut[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nrows(); i++)
                    temp[i][j] = (int)data[cc++];
        }
        return mOut;
    }

    public int[][] to_int2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        int[][] mOut = new int[nrows()][nc];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = (int)data[cc++];

        return mOut;
    }

    public int[] to_int1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * nc * nch;
        int[] vec = new int[ndata_new];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (int)data[ii];
        return vec;
    }

    public byte[][][] to_byte3DArray()
    {
        byte[][][] mOut = new byte[nch][nrows()][nc];
        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            byte[][] temp = mOut[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nrows(); i++)
                    temp[i][j] = (byte)data[cc++];
        }
        return mOut;
    }

    public byte[][] to_byte2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        byte[][] mOut = new byte[nrows()][nc];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = (byte)data[cc++];
        return mOut;
    }

    public byte[] to_byte1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");
        int ndata_new = nrows() * nc * nch;
        byte[] vec = new byte[ndata_new];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (byte)data[ii];
        return vec;
    }

    // convert to Apache Common Math RealMatrix
    public RealMatrix to_ACM_RealMatrix()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        Array2DRowRealMatrix mOut = new Array2DRowRealMatrix(nrows(), nc);
        double[][] m = mOut.getDataRef();

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                m[i][j] = data[cc++];

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
    public static Matkc randi(int nrows, int ncols, int nchannels, int imin, int imax)
    {
        Matkc mOut = new Matkc(nrows, ncols, nchannels);
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
    public static Matkc rand(int nrows, int ncols, int nchannels, double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Matkc mOut = new Matkc(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return mOut;
    }

    public static Matkc rand(int nrows, int ncols, int nchannels)
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
    public static Matkc randn(int nrows, int ncols, int nchannels, double mean, double std)
    {
        Matkc mOut = new Matkc(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = rand.nextGaussian() * std + mean;
        return mOut;
    }

    public static Matkc fill_ladder(int nrows, int ncols, int nchannels, double start_val, double step)
    {
        Matkc mOut = new Matkc(nrows, ncols, nchannels);
        for(int ii=0; ii<mOut.ndata(); ii++)
        {
            mOut.data[ii] = start_val;
            start_val += step;
        }
        return mOut;
    }

    public static Matkc fill_ladder(int nrows, int ncols, int nchannels, double start_val)
    {
        return fill_ladder(nrows, ncols, nchannels, start_val, 1);
    }

    public static Matkc fill_ladder(int nrows, int ncols, int nchannels)
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
    public static Matkc linspace(double start_val, double end_val, int nvals, String vector_type)
    {
        Matkc mOut;

        switch(vector_type)
        {
            case "row":
                mOut = new Matkc(1, nvals, 1);
                break;
            case "col":
                mOut = new Matkc(nvals, 1, 1);
                break;
            case "channel":
                mOut = new Matkc(1, 1, nvals);
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

    public static Matkc linspace(double start_val, double end_val, int nvals)
    {
        return linspace(start_val, end_val, nvals, "row");
    }

    public static Matkc randn(int nrows, int ncols, int nchannels)
    {
        return randn(nrows, ncols, nchannels, 0, 1);
    }

    public static Matkc ones(int nrows, int ncols, int nchannels, double val)
    {
        Matkc mOut = new Matkc(nrows, ncols, nchannels);
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = val;
        return mOut;
    }

    public static Matkc ones(int nrows, int ncols, int nchannels)
    {
        return ones(nrows, ncols, nchannels, 1);
    }

    public static Matkc zeros(int nrows, int ncols, int nchannels)
    {
        return new Matkc(nrows, ncols, nchannels);
    }

    /**
     * Make a deep copy of the current matrix.
     * @return
     */
    public Matkc copy_deep()
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        System.arraycopy( data, 0, mOut.data, 0, ndata );
        return mOut;
    }

    public int nrows()
    {
        return nr;
    }

    public int ncols()
    {
        return nc;
    }

    public int nchannels()
    {
        return nch;
    }

    public int ndata() { return ndata; }

    public int ndata_per_chan() { return ndata_per_chan; }

    public int length_vec()
    {
        if(!is_vector())
            throw new IllegalArgumentException("ERROR: this matrix is not a vector");

        return ndata;
    }

    public double[] data() { return data; }

    /**
     * Find out whether this matrix is a vector (either row, column or channel vector)
     * @return
     */
    public boolean is_vector()
    {
        // a vector is a 3D matrix for which two of the dimensions has length of one.
        int z1 = nr == 1 ? 1:0;
        int z2 = nc == 1 ? 1:0;
        int z3 = nch == 1 ? 1:0;
        return z1+z2+z3 >= 2;
    }

    public boolean is_row_vector()
    {
        return nr == 1 && nc >= 1 && nch == 1;
    }

    public boolean is_col_vector()
    {
        return nr >= 1 && nc == 1 && nch == 1;
    }

    public boolean is_channel_vector()
    {
        return nr == 1 && nc == 1 && nch >= 1;
    }

    // get the copy of data corresponding to given range of a full matrix
    public Matkc get(int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
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

        Matkc mOut = new Matkc(nr_new, nc_new, nch_new);
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
                cc += ndata_per_chan_new;
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

    public Matkc get2(int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
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

        Matkc mOut = new Matkc(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=ch1; k<=ch2; k++)
            for(int j=c1; j<=c2; j++)
                for(int i=r1; i<=r2; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + j * nr + i];

        return mOut;
    }

    public double[] get_arrayOutput(int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
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

        double[] temp_out = new double[nr_new * nc_new * nch_new];

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

        return temp_out;
    }

    public Matkc get(int r1, int r2, int c1, int c2)
    {
        return get(r1, r2, c1, c2, 0, -1);
    }

    public double[] get_arrayOutput(int r1, int r2, int c1, int c2)
    {
        return get_arrayOutput(r1, r2, c1, c2, 0, -1);
    }

    // get an element
    public double get(int i, int j, int k)
    {
        return data[k * ndata_per_chan + j * nr + i];
    }

    // assume k=0
    public double get(int i, int j)
    {
        return data[j * nr + i];
    }

    // get from a linear index
    public double get(int lin_index)
    {
        return data[lin_index];
    }

    // get first element
    public double get()
    {
        return data[0];
    }

    // get a discontinuous submatrix of the current matrix.
    public Matkc get(int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;
        Matkc mOut = new Matkc(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]];

        return mOut;
    }

    public double[] get_arrayOutput(int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;
        double[] temp_out = new double[nr_new * nc_new * nch_new];

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]];

        return temp_out;
    }

    public Matkc get_row(int row_index)
    {
        return get(row_index, row_index, 0, -1, 0, -1);
    }

    public double[] get_row_arrayOutput(int row_index)
    {
        return get_arrayOutput(row_index, row_index, 0, -1, 0, -1);
    }

    public Matkc get_rows(int start_index, int end_index)
    {
        return get(start_index, end_index, 0, -1, 0, -1);
    }

    public double[] get_rows_arrayOutput(int start_index, int end_index)
    {
        return get_arrayOutput(start_index, end_index, 0, -1, 0, -1);
    }

    // take a discontinuous submatrix in the form of rows
    public Matkc get_rows(int[] row_indices)
    {
        int nr_new = row_indices.length;
        Matkc mOut = new Matkc(nr_new, nc, nch);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + j * nr + row_indices[i]];

        return mOut;
    }

    public double[] get_rows_arrayOutput(int[] row_indices)
    {
        int nr_new = row_indices.length;
        double[] temp_out = new double[nr_new * nc * nch];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + j * nr + row_indices[i]];

        return temp_out;
    }

    // take a continuous submatrix in the form of a column
    public Matkc get_col(int col_index)
    {
        return get(0, -1, col_index, col_index, 0, -1);
    }

    public double[] get_col_arrayOutput(int col_index)
    {
        return get_arrayOutput(0, -1, col_index, col_index, 0, -1);
    }

    // take a continuous submatrix in the form of cols
    public Matkc get_cols(int start_index, int end_index)
    {
        return get(0, -1, start_index, end_index, 0, -1);
    }

    public double[] get_cols_arrayOutput(int start_index, int end_index)
    {
        return get_arrayOutput(0, -1, start_index, end_index, 0, -1);
    }

    // take a discontinuous submatrix in the form of cols
    public Matkc get_cols(int[] col_indices)
    {
        int nc_new = col_indices.length;
        Matkc mOut = new Matkc(nr, nc_new, nch);
        double[] temp_out = mOut.data;

//        int cc = 0;
//        for(int k=0; k<nch; k++)
//            for(int j=0; j<nc_new; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[k * ndata_per_chan + col_indices[j] * nr + i];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
            {
                System.arraycopy(data, k * ndata_per_chan + col_indices[j]*nr, temp_out, cc, nr);
                cc += nr;
            }

        return mOut;
    }

    public double[] get_cols_arrayOutput(int[] col_indices)
    {
        int nc_new = col_indices.length;
        double[] temp_out = new double[nr * nc_new * nch];

//        int cc = 0;
//        for(int k=0; k<nch; k++)
//            for(int j=0; j<nc_new; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[k * ndata_per_chan + col_indices[j] * nr + i];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
            {
                System.arraycopy(data, k * ndata_per_chan + col_indices[j]*nr, temp_out, cc, nr);
                cc += nr;
            }

        return temp_out;
    }

    // take a continuous submatrix in the form of a channel
    public Matkc get_channel(int channel_index)
    {
        return get(0, -1, 0, -1, channel_index, channel_index);
    }

    public double[] get_channel_arrayOutput(int channel_index)
    {
        return get_arrayOutput(0, -1, 0, -1, channel_index, channel_index);
    }

    // take a continuous submatrix in the form of channels
    public Matkc get_channels(int start_index, int end_index)
    {
        return get(0, -1, 0, -1, start_index, end_index);
    }

    public double[] get_channels_arrayOutput(int start_index, int end_index)
    {
        return get_arrayOutput(0, -1, 0, -1, start_index, end_index);
    }

    // take a discontinuous submatrix in the form of cols
    public Matkc get_channels(int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        Matkc mOut = new Matkc(nr, nc, nch_new);
        double[] temp_out = mOut.data;

//        int cc = 0;
//        for(int k=0; k<nch_new; k++)
//            for(int j=0; j<nc; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + j * nr + i];

        for(int k=0; k<nch_new; k++)
            System.arraycopy( data, channel_indices[k] * ndata_per_chan, temp_out, k * ndata_per_chan, ndata_per_chan );

        return mOut;
    }

    public double[] get_channels_arrayOutput(int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        double[] temp_out = new double[nr * nc * nch_new];

//        int cc = 0;
//        for(int k=0; k<nch_new; k++)
//            for(int j=0; j<nc; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + j * nr + i];

        for(int k=0; k<nch_new; k++)
            System.arraycopy( data, channel_indices[k] * ndata_per_chan, temp_out, k * ndata_per_chan, ndata_per_chan );

        return temp_out;
    }

    // use the entire given input matrix to set part of this matrix with the given range
    public Matkc set(Matkc mIn, int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
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

        if(nr_new != mIn.nr || nc_new != mIn.nc || nch_new != mIn.nch)
            throw new IllegalArgumentException("ERROR: the input matrix and the range specified do not match.");

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        if(nr_new == nr && nc_new == nc)
        {
            System.arraycopy( temp_in, 0, temp_out, ch1 * ndata_per_chan, ndata_new );
        }
        else if(nr_new == nr && nc_new != nc)
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
            {
                System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + c1*nr, ndata_per_chan_new );
                cc += nr_new * nc_new;
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
                for(int j=0; j<nc_new; j++)
                {
                    System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, nr_new );
                    cc += nr_new;
                }
        }

        return this;
    }

    // use the entire given input matrix (in the form of an
    // array stored in col major order to set part of this matrix with the given range
    public Matkc set(double[] data_mIn, int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
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

        if(ndata_new != data_mIn.length)
            throw new IllegalArgumentException("ERROR: the input matrix data and the range specified do not match.");

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        if(nr_new == nr && nc_new == nc)
        {
            System.arraycopy( temp_in, 0, temp_out, ch1 * ndata_per_chan, ndata_new );
        }
        else if(nr_new == nr && nc_new != nc)
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
            {
                System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + c1*nr, ndata_per_chan_new );
                cc += nr_new * nc_new;
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
                for(int j=0; j<nc_new; j++)
                {
                    System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, nr_new );
                    cc += nr_new;
                }
        }

        return this;
    }

    // use the entire given input matrix to set part of this matrix with the given range
    public Matkc set(Matkc mIn, int r1, int r2, int c1, int c2)
    {
        return set(mIn, r1, r2, c1, c2, 0, -1);
    }

    public Matkc set(double[] data_mIn, int r1, int r2, int c1, int c2)
    {
        return set(data_mIn, r1, r2, c1, c2, 0, -1);
    }

    // use the entire given input matrix to set part of this matrix starting with i,j,k position
    public Matkc set(Matkc mIn, int i, int j, int k)
    {
        return set(mIn, i, i+mIn.nr-1, j, j+mIn.nc-1, k, k+mIn.nch-1);
    }

    // use the entire given input matrix to set part of this matrix starting with i,j,0 position
    public Matkc set(Matkc mIn, int i, int j)
    {
        return set(mIn, i, i+mIn.nr-1, j, j+mIn.nc-1, 0, mIn.nch-1);
    }

    // use the given value to set an element of this matrix at i,j,k position
    public Matkc set(double val, int i, int j, int k)
    {
        data[k * ndata_per_chan + j * nr + i] = val;
        return this;
    }

    // use the given value to set an element of this matrix at i,j,0 position
    public Matkc set(double val, int i, int j)
    {
        data[j * nr + i] = val;
        return this;
    }

    // use the given value to set an element of this matrix at lin_index linear position
    public Matkc set(double val, int lin_index)
    {
        data[lin_index] = val;
        return this;
    }

    // use the given value to set an element of this matrix at 0,0,0 position
    public Matkc set(double val)
    {
        data[0] = val;
        return this;
    }

    // use the entire given input matrix to set part of this matrix specified by row, col and chan indices.
    public Matkc set(Matkc mIn, int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    public Matkc set(double[] data_mIn, int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    // use the entire given input matrix (must be a row vector) to set a row of this matrix
    public Matkc set_row(Matkc mIn, int row_index)
    {
        return set(mIn, row_index, row_index, 0, -1, 0, -1);
    }

    public Matkc set_row(double[] data_mIn, int row_index)
    {
        return set(data_mIn, row_index, row_index, 0, -1, 0, -1);
    }

    // use the entire given input matrix to a range of rows of this matrix
    public Matkc set_rows(Matkc mIn, int start_index, int end_index)
    {
        return set(mIn, start_index, end_index, 0, -1, 0, -1);
    }

    public Matkc set_rows(double[] data_mIn, int start_index, int end_index)
    {
        return set(data_mIn, start_index, end_index, 0, -1, 0, -1);
    }

    // use the entire given input matrix to set specified rows of this matrix.
    public Matkc set_rows(Matkc mIn, int[] row_indices)
    {
        int nr_new = row_indices.length;

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[k * ndata_per_chan + j * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    public Matkc set_rows(double[] data_mIn, int[] row_indices)
    {
        int nr_new = row_indices.length;

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[k * ndata_per_chan + j * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    // use the entire given input matrix (must be a col vector) to set a col of this matrix
    public Matkc set_col(Matkc mIn, int col_index)
    {
        return set(mIn, 0, -1, col_index, col_index, 0, -1);
    }

    public Matkc set_col(double[] data_mIn, int col_index)
    {
        return set(data_mIn, 0, -1, col_index, col_index, 0, -1);
    }

    // use the entire given input matrix to a range of cols of this matrix
    public Matkc set_cols(Matkc mIn, int start_index, int end_index)
    {
        return set(mIn, 0, -1, start_index, end_index, 0, -1);
    }

    public Matkc set_cols(double[] data_mIn, int start_index, int end_index)
    {
        return set(data_mIn, 0, -1, start_index, end_index, 0, -1);
    }

    // use the entire given input matrix to set specified cols of this matrix.
    public Matkc set_cols(Matkc mIn, int[] col_indices)
    {
        int nc_new = col_indices.length;

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr; i++)
                    temp_out[k * ndata_per_chan + col_indices[j] * nr + i] = temp_in[cc++];

        return this;
    }

    public Matkc set_cols(double[] data_mIn, int[] col_indices)
    {
        int nc_new = col_indices.length;

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr; i++)
                    temp_out[k * ndata_per_chan + col_indices[j] * nr + i] = temp_in[cc++];

        return this;
    }

    // use the entire given input matrix (must be a channel) to set a channel of this matrix
    public Matkc set_channel(Matkc mIn, int channel_index)
    {
        return set(mIn, 0, -1, 0, -1, channel_index, channel_index);
    }

    public Matkc set_channel(double[] data_mIn, int channel_index)
    {
        return set(data_mIn, 0, -1, 0, -1, channel_index, channel_index);
    }

    // use the entire given input matrix to a range of channels of this matrix
    public Matkc set_channels(Matkc mIn, int start_index, int end_index)
    {
        return set(mIn, 0, -1, 0, -1, start_index, end_index);
    }

    public Matkc set_channels(double[] data_mIn, int start_index, int end_index)
    {
        return set(data_mIn, 0, -1, 0, -1, start_index, end_index);
    }


    // use the entire given input matrix to set specified channels of this matrix.
    public Matkc set_channels(Matkc mIn, int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + j * nr + i] = temp_in[cc++];

        return this;
    }

    public Matkc set_channels(double[] data_mIn, int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + j * nr + i] = temp_in[cc++];

        return this;
    }

    public void print(String name_matrix)
    {
        System.out.println("=========== Printing matrix ===========");
        for(int k=0; k<nch; k++)
        {
            System.out.println(name_matrix + "(:,:," + (k+1) + ")=[");
            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc-1; j++)
                    System.out.print(get(i,j,k) + ",\t");
                System.out.println(get(i,nc-1,k) + ";");
            }
            System.out.println("];");
        }
        System.out.println("=========== Matrix printed ===========");
    }

    public void print()
    {
        print("mat");
    }

    public void print_info()
    {
        System.out.format("Matrix info: #rows = %d, #cols = %d, #channels = %d.\n", nr, nc, nch);
    }

    public void print_info(String name_matrix)
    {
        System.out.format("Matrix %s info: #rows = %d, #cols = %d, #channels = %d.\n", name_matrix, nr, nc, nch);
    }

    /**
     * Flatten the current matrix to either a row, column or channel vector matrix
     * Always results in a copy.
     * @param target_vec Can be "row", "column" or "channel".
     *                   If row, will result in a row vector.
     *                   If column, will result in a col vector, etc.
     * @return
     */
    public Matkc vectorize(String target_vec)
    {
        Matkc mOut;

        switch(target_vec)
        {
            case "row":
                mOut = new Matkc(1, ndata, 1);
                break;
            case "column":
                mOut = new Matkc(ndata, 1, 1);
                break;
            case "channel":
                mOut = new Matkc(1, 1, ndata);
                break;
            default:
                throw new IllegalArgumentException("ERROR: target_vec must be either \"row\", \"column\" or \"channel\".");
        }

        System.arraycopy( data, 0, mOut.data, 0, ndata );
        return mOut;
    }

    /**
     * Similar as vectorize() but returns an array instead of Matkc
     * @return
     */
    public double[] vectorize_to_doubleArray()
    {
        double[] vec = new double[ndata];
        System.arraycopy( data, 0, vec, 0, ndata );
        return vec;
    }

    public float[] vectorize_to_floatArray()
    {
        float[] vec = new float[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (float)data[ii];
        return vec;
    }

    public int[] vectorize_to_intArray()
    {
        int[] vec = new int[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (int)data[ii];
        return vec;
    }

    public byte[] vectorize_to_byteArray()
    {
        byte[] vec = new byte[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (byte)data[ii];
        return vec;
    }

    public Double[] vectorize_to_DoubleArray()
    {
        Double[] vec = new Double[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = data[ii];
        return vec;
    }

    public Float[] vectorize_to_FloatArray()
    {
        Float[] vec = new Float[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (float)data[ii];
        return vec;
    }

    public Integer[] vectorize_to_IntegerArray()
    {
        Integer[] vec = new Integer[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (int)data[ii];
        return vec;
    }

    public Byte[] vectorize_to_ByteArray()
    {
        Byte[] vec = new Byte[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (byte)data[ii];
        return vec;
    }

    /**
     * Transpose this matrix
     * @return tranposed matrix
     */
    public Matkc t()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: Cannot transpose matrix with more than 1 channel");

        Matkc mOut = new Matkc(nc, nr, 1);
        double[] temp_out = mOut.data;
        int cc = 0;
        for(int i=0; i<nr; i++)
            for(int j=0; j<nc; j++)
                temp_out[cc++] = data[j * nr + i];
        return mOut;
    }

    // reverse the channels in the matrix
    // this is useful for converting from RGB to BGR channels and vice-versa
    public Matkc reverse_channels()
    {
        int[] indices_channels = new int[nch];
        for(int i=nch-1, j=0; i>=0; i--, j++)
            indices_channels[j] = i;
        return get_channels(indices_channels);
    }

    public Matkc increment_IP()
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] + 1;
        return this; // just for convenience
    }

    public Matkc increment()
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] + 1;
        return mOut;
    }

    public Matkc decrement_IP()
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] - 1;
        return this; // just for convenience
    }

    public Matkc decrement()
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] - 1;
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
        if (!(obj_ instanceof Matkc)) {
            return false;
        }

        // typecast o to Complex so that we can compare data members
        Matkc mIn = (Matkc) obj_;

        if( (mIn.nr != nr ) || (mIn.nc != nc ) ||
                (mIn.nch != nch ))
            return false;

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
        {
            if(data[ii] != temp_in[ii])
                return false;
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
    public boolean equals_approx(Matkc mIn, double tolerance) {

        // If the object is compared with itself then return true
        if (mIn == this) {
            return true;
        }

        if( (mIn.nr != nr ) || (mIn.nc != nc ) ||
                (mIn.nch != nch ))
            return false;

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
        {
            if(Math.abs(data[ii] - temp_in[ii]) > tolerance)
                return false;
        }

        return true;
    }

    /**
     * perform dot product between this matrix and given matrix.
     * Assume that the given matrices are column or row vectors.
     * @param mIn
     * @return
     */
    public double dot(Matkc mIn)
    {
        if(ndata != mIn.ndata)
            throw new IllegalArgumentException("ERROR: ndata != mIn.ndata");

        double sum = 0;
        double temp[] = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            sum += (data[ii] * temp[ii]);

        return sum;
    }

    /**
     * element-wise multiplication of two matrices
     * @param mIn
     * @return
     */
    public Matkc multE(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_in = mIn.data;
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] * temp_in[ii]);

        return mOut;
    }

    public Matkc multE_IP(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] * temp_in[ii]);

        return this;
    }

    public Matkc mult(double val)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] * val;

        return mOut;
    }

    public Matkc mult_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] * val;
        return this;
    }

    /**
     * Multiply this matrix with another matrix.
     * @param mIn
     * @return
     */
    public Matkc mult(Matkc mIn)
    {
        if(nch != 1 || mIn.nch != 1)
            throw new IllegalArgumentException("ERROR: matrix multiplication can be performed on matrices with one channel.");

        if( nc != mIn.nr )
            throw new IllegalArgumentException("ERROR: Invalid sizes of matrices for mutiplication.");

        int nr_new = nr;
        int nc_new = mIn.nc;

        Matkc mOut = new Matkc(nr_new, nc_new, 1);
        double[] temp_out = mOut.data;
        int cc = 0;
        for(int j=0; j<nc_new; j++)
            for(int i=0; i<nr_new; i++)
                temp_out[cc++] = get_row(i).dot(mIn.get_col(j));

        return mOut;
    }

    public Matkc divE(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] / temp_in[ii]);
        return mOut;
    }

    public Matkc divE_IP(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] / temp_in[ii]);
        return this;
    }

    public Matkc div(double val)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] / val;
        return mOut;
    }

    public Matkc div_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] / val;
        return this;
    }

    public Matkc pow(double val)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.pow(data[ii], val);
        return mOut;
    }

    public Matkc pow_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.pow(data[ii], val);
        return this;
    }

    public Matkc plus(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot add two matrices of different sizes.");

        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] + temp_in[ii]);
        return mOut;
    }

    public Matkc plus_IP(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot add two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] + temp_in[ii]);
        return this;
    }

    public Matkc plus(double val)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] + val;
        return mOut;
    }

    public Matkc plus_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] + val;
        return this;
    }

    public Matkc minus(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot subtract two matrices of different sizes.");

        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] - temp_in[ii]);
        return mOut;
    }

    public Matkc minus_IP(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot subtract two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] - temp_in[ii]);
        return this;
    }

    public Matkc minus(double val)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] - val;
        return mOut;
    }

    public Matkc minus_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] - val;
        return this;
    }

    // Generate a matrix by replicating this matrix in a block-like fashion
    // similar to matlab's repmat
    public Matkc repmat(int ncopies_row, int ncopies_col, int ncopies_ch)
    {
        int nrows_this = nr;
        int ncols_this = nc;
        int nchannels_this = nch;

        Matkc matOut = new Matkc(nrows_this*ncopies_row,
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
                    matOut.set(this, row1, row2, col1, col2, chan1, chan2);
                }

        return matOut;
    }

    /**
     * Reshape a matrix.
     * @param nrows_new
     * @param ncols_new
     * @param nchannels_new
     */
    public Matkc reshape(int nrows_new, int ncols_new, int nchannels_new)
    {
        if (nrows_new * ncols_new * nchannels_new != ndata)
            throw new IllegalArgumentException("ERROR: nrows_new * ncols_new * nchannels_new != ndata.");

        Matkc mOut = new Matkc(nrows_new, ncols_new, nchannels_new);
        System.arraycopy( data, 0, mOut.data, 0, ndata );
        return mOut;
    }

    public Matkc round()
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.round(data[ii]);
        return mOut;
    }

    public Matkc round_IP()
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.round(data[ii]);
        return this;
    }

    public Matkc zeros_IP()
    {
        Arrays.fill(data, 0);
        return this;
    }

    public Matkc ones_IP()
    {
        Arrays.fill(data, 1);
        return this;
    }

    public Matkc fill_IP(double val)
    {
        Arrays.fill(data, val);
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
    public Matkc randi_IP(int imin, int imax)
    {
        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (double)(rand.nextInt(imax + 1 - imin) + imin);
        return this;
    }

    public Matkc randi(int imin, int imax)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
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
    public Matkc rand(double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Matkc mOut = new Matkc(nr, nc, nch);
        Random rand = new Random();
        double[] temp_out = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp_out[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return mOut;
    }

    public Matkc rand()
    {
        return rand(0, 1);
    }

    public Matkc rand_IP(double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return this;
    }

    public Matkc rand_IP()
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
    public Matkc randn(double mean, double std)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        Random rand = new Random();
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp[ii] = rand.nextGaussian() * std + mean;
        return mOut;
    }

    public Matkc randn()
    {
        return randn(0, 1);
    }

    public Matkc randn_IP(double mean, double std)
    {
        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = rand.nextGaussian() * std + mean;
        return this;
    }

    public Matkc randn_IP()
    {
        return randn_IP(0, 1);
    }

    public Matkc rand_custom(DoubleSupplier functor)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        Random rand = new Random();
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp[ii] = functor.getAsDouble();
        return mOut;
    }

    public Matkc rand_custom_IP(DoubleSupplier functor)
    {
        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = functor.getAsDouble();
        return this;
    }

    public Matkc fill_ladder(double start_val, double step)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
        {
            temp[ii] = start_val;
            start_val += step;
        }
        return mOut;
    }

    public Matkc fill_ladder_IP(double start_val, double step)
    {
        for(int ii=0; ii<ndata; ii++)
        {
            data[ii] = start_val;
            start_val += step;
        }
        return this;
    }

    public Result_sort sort(boolean sort_col, boolean sort_ascend)
    {
        if(nch!=1)
            throw new IllegalArgumentException("ERROR: for sorting, this matrix must have only one channel.");

        int number_rows = nr;
        int number_cols = nc;

        Result_sort res = new Result_sort();

        res.matSorted = new Matkc(number_rows, number_cols, 1);
        res.indices_sort = new Matkc(number_rows, number_cols, 1);

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

    public Matkc max(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: This matrix and input matrix do not have same size.");

        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.max(data[ii], temp_in[ii]);
        return mOut;
    }

    public Matkc max_IP(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise divide two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.max(data[ii], temp_in[ii]);
        return this;
    }

    public Matkc max(double val)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.max(data[ii], val);
        return mOut;
    }

    public Matkc max_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.max(data[ii], val);
        return this;
    }

    public Result_minMax_eachDim max(String process_dim)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

        switch(process_dim)
        {
            case "col":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matkc(1, number_cols, number_chans);
                res.matIndices = new Matkc(1, number_cols, number_chans);
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
                res.matVals = new Matkc(number_rows, 1, number_chans);
                res.matIndices = new Matkc(number_rows, 1, number_chans);
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
                res.matVals = new Matkc(number_rows, number_cols, 1);
                res.matIndices = new Matkc(number_rows, number_cols, 1);
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
        res.val = data[0];
        res.i = 0;
        res.j = 0;
        res.k = 0;
        double val_cur;

        int cc = 0;

        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                {
                    val_cur = data[cc++];
                    if(val_cur > res.val)
                    {
                        res.val = val_cur;
                        res.i = i;
                        res.j = j;
                        res.k = k;
                    }
                }

        return res;
    }

    public Matkc min(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: This matrix and input matrix do not have same size.");

        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.min(data[ii], temp_in[ii]);
        return mOut;
    }

    public Matkc min_IP(Matkc mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise divide two matrices of different sizes.");
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.min(data[ii], temp_in[ii]);
        return this;
    }

    public Matkc min(double val)
    {
        Matkc mOut = new Matkc(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.min(data[ii], val);
        return mOut;
    }

    public Matkc min_IP(double val)
    {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.min(data[ii], val);
        return this;
    }

    public Result_minMax_eachDim min(String process_dim)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

        switch(process_dim)
        {
            case "col":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matkc(1, number_cols, number_chans);
                res.matIndices = new Matkc(1, number_cols, number_chans);
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
                res.matVals = new Matkc(number_rows, 1, number_chans);
                res.matIndices = new Matkc(number_rows, 1, number_chans);
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
                res.matVals = new Matkc(number_rows, number_cols, 1);
                res.matIndices = new Matkc(number_rows, number_cols, 1);
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
        res.val = data[0];
        res.i = 0;
        res.j = 0;
        res.k = 0;
        double val_cur;

        int cc = 0;

        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                {
                    val_cur = data[cc++];
                    if(val_cur < res.val)
                    {
                        res.val = val_cur;
                        res.i = i;
                        res.j = j;
                        res.k = k;
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
    public Matkc moment(String moment_type, String process_dim, boolean isBiasCorrected)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

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
                Matkc matOut = new Matkc(1, number_cols, number_chans);
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
                Matkc matOut = new Matkc(number_rows, 1, number_chans);
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
                Matkc matOut = new Matkc(number_rows, number_cols, 1);
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

    public Matkc moment(String moment_type, String process_dim)
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
    public Matkc summarize(String process_dim, functor_double_doubleArray functor)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

        switch(process_dim)
        {
            case "col":
            {
                Matkc matOut = new Matkc(1, number_cols, number_chans);
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
                Matkc matOut = new Matkc(number_rows, 1, number_chans);
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
                Matkc matOut = new Matkc(number_rows, number_cols, 1);
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

    public Matkc median(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.percentile(ee, 0.5));
    }

    public Matkc percentile(String process_dim, double p)
    {
        return summarize(process_dim, ee->StatUtils.percentile(ee, p));
    }

    public Matkc mode(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.mode(ee)[0]);
    }

    public Matkc product(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.product(ee));
    }

    // sum all the elements in this matrix
    public double sum()
    {
        double total = 0;
        for(int ii=0; ii<ndata; ii++)
            total += data[ii];
        return total;
    }

    public Matkc sum(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sum(ee));
    }

    // Returns the sum of the natural logs of the entries in the input array, or Double.NaN if the array is empty.
    public Matkc sumLog(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sumLog(ee));
    }

    //Returns the sum of the squares of the entries in the input array, or Double.NaN if the array is empty.
    public Matkc sumSq(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sumSq(ee));
    }

    //Returns the variance of the entries in the input array, or Double.NaN if the array is empty.
    // population version = true means  ( sum((x_i - mean)^2) / n )
    // population version = false means  ( sum((x_i - mean)^2) / (n-1) )
    public Matkc variance(String process_dim, boolean population_version)
    {
        if(population_version)
            return summarize(process_dim, ee->StatUtils.populationVariance(ee));
        else
            return summarize(process_dim, ee->StatUtils.variance(ee));
    }

    //Returns the variance of the entries in the input array, or Double.NaN if the array is empty.
    // ( sum((x_i - mean)^2) / (n-1) )
    public Matkc variance(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.variance(ee));
    }

    //Returns the variance of the entries in the input array, using the precomputed mean value.
    // population version = true means  ( sum((x_i - mean)^2) / n )
    // population version = false means  ( sum((x_i - mean)^2) / (n-1) )
    public Matkc variance(String process_dim, double mean, boolean population_version)
    {
        if(population_version)
            return summarize(process_dim, ee->StatUtils.populationVariance(ee, mean));
        else
            return summarize(process_dim, ee->StatUtils.variance(ee, mean));
    }

    //Returns the variance of the entries in the input array, using the precomputed mean value.
    //( sum((x_i - mean)^2) / (n-1) )
    public Matkc variance(String process_dim, double mean)
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

        int cc = 0;

        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                {
                    curVal = data[cc++];
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
    public Matkc add_cols(Matkc matIn)
    {
        int nrows_new = Math.max(nr, matIn.nr);
        int ncols_new = nc + matIn.nc;
        int nch_new = Math.max(nch, matIn.nch);
        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        matOut.set(this, 0, nr-1, 0, nc-1, 0, nch-1);
        matOut.set(matIn, 0, matIn.nr-1, nc, ncols_new-1, 0, matIn.nch-1);
        return matOut;
    }

    // merge an array of matrices horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matkc merge_cols(Matkc[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat[kk].nr);
            ncols_new += vmat[kk].nc;
            nch_new = Math.max(nch_new, vmat[kk].nch);
        }

        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        int nc_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat[kk], 0, vmat[kk].nr - 1, nc_count, nc_count + vmat[kk].nc - 1, 0, vmat[kk].nch - 1);
            nc_count += vmat[kk].nc;
        }

        return matOut;
    }

    // merge a list of matrices horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matkc merge_cols(List<Matkc> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat.get(kk).nr);
            ncols_new += vmat.get(kk).nc;
            nch_new = Math.max(nch_new, vmat.get(kk).nch);
        }

        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        int nc_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat.get(kk), 0, vmat.get(kk).nr - 1, nc_count, nc_count + vmat.get(kk).nc - 1, 0, vmat.get(kk).nch - 1);
            nc_count += vmat.get(kk).nc;
        }

        return matOut;
    }

    // join this matrix with the given matrix matIn vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public Matkc add_rows(Matkc matIn)
    {
        int nrows_new = nr + matIn.nr;
        int ncols_new = Math.max(nc, matIn.nc);
        int nch_new = Math.max(nch, matIn.nch);
        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        matOut.set(this, 0, nr - 1, 0, nc - 1, 0, nch - 1);
        matOut.set(matIn, nr, nrows_new - 1, 0, matIn.nc - 1, 0, matIn.nch - 1);
        return matOut;
    }

    // merge an array of matrices vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matkc merge_rows(Matkc[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new += vmat[kk].nr;
            ncols_new = Math.max(ncols_new, vmat[kk].nc);
            nch_new = Math.max(nch_new, vmat[kk].nch);
        }

        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        int nr_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat[kk], nr_count, nr_count + vmat[kk].nr - 1, 0, vmat[kk].nc - 1, 0, vmat[kk].nch - 1);
            nr_count += vmat[kk].nr;
        }

        return matOut;
    }

    // merge a list of matrices vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matkc merge_rows(List<Matkc> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new += vmat.get(kk).nr;
            ncols_new = Math.max(ncols_new, vmat.get(kk).nc);
            nch_new = Math.max(nch_new, vmat.get(kk).nch);
        }

        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        int nr_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat.get(kk), nr_count, nr_count + vmat.get(kk).nr - 1, 0, vmat.get(kk).nc - 1, 0, vmat.get(kk).nch - 1);
            nr_count += vmat.get(kk).nr;
        }

        return matOut;
    }

    // add channels to the this matrix.
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public Matkc add_channels(Matkc matIn)
    {
        int nrows_new = Math.max(nr, matIn.nr);
        int ncols_new = Math.max(nc, matIn.nc);
        int nch_new = nch + matIn.nch;
        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        matOut.set(this, 0, nr - 1, 0, nc - 1, 0, nch - 1);
        matOut.set(matIn, 0, matIn.nr - 1, 0, matIn.nc - 1, nch, nch_new - 1);
        return matOut;
    }

    // merge channels of an array of matrices
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public static Matkc merge_channels(Matkc[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat[kk].nr);
            ncols_new = Math.max(ncols_new, vmat[kk].nc);
            nch_new += vmat[kk].nch;
        }

        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        int nch_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat[kk], 0, vmat[kk].nr - 1, 0, vmat[kk].nc - 1, nch_count, nch_count + vmat[kk].nch - 1);
            nch_count += vmat[kk].nch;
        }

        return matOut;
    }

    // merge channels of a list of matrices
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public static Matkc merge_channels(List<Matkc> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat.get(kk).nr);
            ncols_new = Math.max(ncols_new, vmat.get(kk).nc);
            nch_new += vmat.get(kk).nch;
        }

        Matkc matOut = new Matkc(nrows_new, ncols_new, nch_new);
        int nch_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat.get(kk), 0, vmat.get(kk).nr - 1, 0, vmat.get(kk).nc - 1, nch_count, nch_count + vmat.get(kk).nch - 1);
            nch_count += vmat.get(kk).nch;
        }

        return matOut;
    }

    // remove cols from this matrix
    public Matkc del_cols(int[] indices_remove)
    {
        // row indices that I want to keep (keep all; 1,2,...,nr)
        int[] row_idxs_keep = stdfuncs.fill_ladder_int(nr, 0, 1);

        // channel indices that I want to keep (keep all; 1,2,...,nch)
        int[] ch_idxs_keep = stdfuncs.fill_ladder_int(nch, 0, 1);

        // col indices to keep
        int[] col_idxs_all = stdfuncs.fill_ladder_int(nc, 0, 1);
        int[] col_idxs_keep = stdfuncs.set_diff(col_idxs_all, indices_remove);

        Matkc matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove rows from this matrix
    public Matkc del_rows(int[] indices_remove)
    {
        // col indices that I want to keep (keep all; 1,2,...,nr)
        int[] col_idxs_keep = stdfuncs.fill_ladder_int(nc, 0, 1);

        // channel indices that I want to keep (keep all; 1,2,...,nch)
        int[] ch_idxs_keep = stdfuncs.fill_ladder_int(nch, 0, 1);

        // row indices to keep
        int[] row_idxs_all = stdfuncs.fill_ladder_int(nr, 0, 1);
        int[] row_idxs_keep = stdfuncs.set_diff(row_idxs_all, indices_remove);

        Matkc matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove channels from this matrix
    public Matkc del_channels(int[] indices_remove)
    {
        // row indices that I want to keep (keep all; 1,2,...,nr)
        int[] row_idxs_keep = stdfuncs.fill_ladder_int(nr, 0, 1);

        // col indices that I want to keep (keep all; 1,2,...,nr)
        int[] col_idxs_keep = stdfuncs.fill_ladder_int(nc, 0, 1);

        // channel indices to keep
        int[] ch_idxs_all = stdfuncs.fill_ladder_int(nch, 0, 1);
        int[] ch_idxs_keep = stdfuncs.set_diff(ch_idxs_all, indices_remove);

        Matkc matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove submatrix from this matrix
    public Matkc del_submat(int[] row_indices_remove, int[] col_indices_remove, int[] channel_indices_remove)
    {
        // row indices to keep
        int[] row_idxs_all = stdfuncs.fill_ladder_int(nr, 0, 1);
        int[] row_idxs_keep = stdfuncs.set_diff(row_idxs_all, row_indices_remove);

        // col indices to keep
        int[] col_idxs_all = stdfuncs.fill_ladder_int(nc, 0, 1);
        int[] col_idxs_keep = stdfuncs.set_diff(col_idxs_all, col_indices_remove);

        // channel indices to keep
        int[] ch_idxs_all = stdfuncs.fill_ladder_int(nch, 0, 1);
        int[] ch_idxs_keep = stdfuncs.set_diff(ch_idxs_all, channel_indices_remove);

        Matkc matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // find the locations of the elements in this matrix that satisfied
    // given number comparison condition
    public Result_find find(String comp_operator, double val)
    {
        int ini_capacity = Math.max(Math.max(nr, nc), nch);
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
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
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
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
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
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
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
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
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
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
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
        if(nch!=1)
            throw new IllegalArgumentException("ERROR: this matrix has more than one channel.");
        KMeansPlusPlusClusterer kmObj = new KMeansPlusPlusClusterer(nclusters, nMaxIters);
        List<vecPointForACMCluster_Matkc> tdata = new ArrayList<vecPointForACMCluster_Matkc>(nc);
        for(int j=0; j<nc; j++)
            tdata.add(new vecPointForACMCluster_Matkc(get_col(j)));
        List<CentroidCluster<vecPointForACMCluster_Matkc>> res = kmObj.cluster(tdata);

        Result_clustering res_cluster = new Result_clustering();

        res_cluster.centroids = new Matkc(nr, nclusters);
        res_cluster.labels = new Matkc(1, nc);
        res_cluster.nclusters = res.size();

        for(int j=0; j<res_cluster.nclusters; j++)
            res_cluster.centroids.set_col(res.get(j).getCenter().getPoint(), j);

        Matkc dists = new Matkc(1, res_cluster.nclusters);
        double dist;

        for(int j=0; j<nc; j++)
        {
            Matkc cur_datapoint = get_col(j);
            // for this data point index j, compute euclidean distance to each of the
            // centroids and save it in dists
            for(int i=0; i<res_cluster.nclusters; i++)
            {
                dist = cur_datapoint.minus(res_cluster.centroids.get_col(i)).pow(2).sum();
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
        res.dataset = new Matkc(2, nclasses * ndata_per_class);
        res.labels = new Matkc(1, nclasses * ndata_per_class);

        for(int i=0; i<nclasses; i++)
        {
            mean[0] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
            mean[1] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
            MultivariateNormalDistribution dist = new MultivariateNormalDistribution(mean, covariance);
            int idx_col;
            for(int j=0; j<ndata_per_class; j++)
            {
                idx_col = (i*ndata_per_class)+j;
                res.dataset.set_col(dist.sample(), idx_col);
                res.labels.set(i+1, idx_col);
            }
        }

        return res;
    }

    // 0.2989 * R + 0.5870 * G + 0.1140 * B
    public Matkc rgb2gray()
    {
        if (nch != 3)
            throw new IllegalArgumentException("ERROR: The input matrix must have 3 channels.");

        Matkc mOut = new Matkc(nr, nc, 1);
        double[] ptr_out = mOut.data;

        for (int ii = 0; ii < ndata_per_chan; ii++)
            ptr_out[ii] = 0.2989 * data[ii] + 0.5870 * data[ndata_per_chan + ii] + 0.1140 * data[ndata_per_chan * 2 + ii];

        return mOut;
    }

    // 0.2989 * R + 0.5870 * G + 0.1140 * B
    public Matkc bgr2gray()
    {
        if (nch != 3)
            throw new IllegalArgumentException("ERROR: The input matrix must have 3 channels.");

        Matkc mOut = new Matkc(nr, nc, 1);
        double[] ptr_out = mOut.data;

        for (int ii = 0; ii < ndata_per_chan; ii++)
            ptr_out[ii] = 0.1140 * data[ii] + 0.5870 * data[ndata_per_chan + ii] +  0.2989 * data[ndata_per_chan * 2 + ii];

        return mOut;
    }

    // normalize dataset using pnorm
    // treat each col of the matrix of a data point
    // for each data point (vector), divide all the elements
    // of the vector by SUM(ABS(V).^P)^(1/P)
    // Note: modify this matrix, just return a reference
    public Matkc normalize_dataset_pNorm(double p)
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: The number of channels of this matrix must be one.");

        double s;
        double tol = 0.00001; // just in case of division by zero
        double p_inv = 1.0 / p;
        double[] v;

        for(int j=0; j<nc; j++)
        {
            v = get_col_arrayOutput(j);
            s = 0;
            for(int i=0; i<nr; i++)
                s += (Math.pow(Math.abs(v[i]), p));
            s = Math.pow(s, p_inv);
            for(int i=0; i<nr; i++)
                v[i] = v[i] / (s + tol);
            set_col(v, j);
        }
        return this;
    }

    public Matkc normalize_dataset_L2Norm()
    {
        return normalize_dataset_pNorm(2.0);
    }

    public Matkc normalize_dataset_L1Norm()
    {
        return normalize_dataset_pNorm(1.0);
    }

}


class vecPointForACMCluster_Matkc implements Clusterable
{
    double[] data;

    vecPointForACMCluster_Matkc(Matkc m)
    {
        data = m.vectorize_to_doubleArray();
    }

    @Override
    public double[] getPoint() {
        return data;
    }

}
