package com.gjs.facelibrary;

/**
 * Created by gjs on 17-10-18.
 */

public class Info {
    public Info() {
    }

    public int rotation;  //0, 90, 180 or 270

    public byte[] yuv; // yuv
    public int width; // preview width
    public int height; // preview height
    public int[] out; // rgba

    public int getRotation() {
        return rotation;
    }

    public void setRotation(int rotation) {
        this.rotation = rotation;
    }

    public byte[] getYuv() {
        return yuv;
    }

    public void setYuv(byte[] yuv) {
        this.yuv = yuv;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int[] getOut() {
        return out;
    }

    public void setOut(int[] out) {
        this.out = out;
    }
}
