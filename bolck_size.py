from ultralytics.nn.modules.block import HGStem, HGBlock, DWConv, Positioning, Focusing
import torch
if __name__ == '__main__':
    intput = torch.randn(1, 512, 40, 40)
    input = torch.randn(1, 256, 40, 40)
    focus = Focusing()

    # HGStem_1 = HGStem(3, 32, 48)
    # output = HGStem_1(intput)
    # print("1.", end='')
    # print(output.size())
    #
    # HGBlock_1 = HGBlock(48,48,128,3,6)
    # output_1 = HGBlock_1(output)
    # print("2.", end='')
    # print(output_1.size())
    #
    # DWConv_1 = DWConv(128, 128, 3, 2, 1, False)
    # output_2 = DWConv_1(output_1)
    # print("3.", end='')
    # print(output_2.size())
    #
    # HGBlock_2 = HGBlock(128, 96, 512, 3, 6)
    # output_3 = HGBlock_2(output_2)
    # print("4.", end='')
    # print(output_3.size())
    #
    # DWConv_2 = DWConv(512, 512, 3, 2, 1, False)
    # output_4 = DWConv_2(output_3)
    # print("5.", end='')
    # print(output_4.size())
    #
    # HGBlock_3 = HGBlock(512, 192, 1024, 5, 6, True, False)
    # output_5 = HGBlock_3(output_4)
    # print("6.", end='')
    # print(output_5.size())
    #
    # HGBlock_4 = HGBlock(1024, 192, 1024, 5, 6, True, True)
    # output_6 = HGBlock_4(output_5)
    # print("7.", end='')
    # print(output_6.size())
    #
    # HGBlock_5 = HGBlock(1024, 192, 1024, 5, 6, True, True)
    # output_7 = HGBlock_5(output_6)
    # print("8.", end='')
    # print(output_7.size())
    #
    # DWConv_3 = DWConv(1024, 1024, 3, 2, 1, False)
    # output_8 = DWConv_3(output_7)
    # print("9.", end='')
    # print(output_8.size())
    #
    # HGBlock_6 = HGBlock(1024, 384, 2048, 5, 6, lightconv=True, shortcut=False)
    # output_9 = HGBlock_6(output_8)
    # print("10.", end='')
    # print(output_9.size())
    #
    # Positioning = Positioning(2048)
    # output_10 = Positioning(output_9)
    # print("11.", end='')
    # print(output_10.size())





