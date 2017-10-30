
/**
 * Created by gjs on 17-10-16.
 */


package com.gjs.facelibrary;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class Face {

    static {
        System.loadLibrary("face");
    }

    // native c++ 方法
    private static native void getLandmark(int rotation, byte[] yuv420sp, int width, int height, int[] rgbOut);
//    private static native void dlibGetLandmark(int rotation, byte[] yuv420sp, int width, int height, int[] rgbOut);
    private native void faceInit(String modelDir);

    // java
    public static int faceDeal(Info info){
        getLandmark(info.rotation, info.yuv, info.width, info.height, info.out);

        return 0;
    };

    /**
     * 初始化方法
     */
    public void init(Context context){
        // 复制 模型 到可访问的存储
        String assetPath = "model";
        String sdcardPath = Environment.getExternalStorageDirectory()
                + File.separator + assetPath;
        copyFilesFromAssets(context, assetPath, sdcardPath);

        // 初始化 jni 加载模型
        faceInit(sdcardPath);
    }

    /**
     * 拷贝资源到 native 可访问的路径
     * @param context
     * @param oldPath
     * @param newPath
     */
    private void copyFilesFromAssets(Context context, String oldPath, String newPath) {
//        Log.i("gjs", "copyFilesFromAssets");
        try {
            String[] fileNames = context.getAssets().list(oldPath);
            if (fileNames.length > 0) {
                // directory
                File file = new File(newPath);
                file.mkdirs();
                // copy recursively
                for (String fileName : fileNames) {
                    copyFilesFromAssets(context, oldPath + "/" + fileName,
                            newPath + "/" + fileName);
                }
            } else {
                // file
                Log.i("gjs copy ", oldPath + " -> " + newPath);
                InputStream is = context.getAssets().open(oldPath);

                FileOutputStream fos = new FileOutputStream(new File(newPath));
                byte[] buffer = new byte[1024];
                int byteCount;
                while ((byteCount = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, byteCount);
                }

                fos.flush();
                is.close();
                fos.close();
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
