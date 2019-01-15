#include "Caffe2Parser.hpp"
//#include "RecordByRecordCaffeParser.hpp"
#include "armnnCaffe2Parser/ICaffe2Parser.hpp"
#include "armnn/Descriptors.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Utils.hpp"
#include "armnn/Exceptions.hpp"

#include "GraphTopologicalSort.hpp"
#include "VerificationHelpers.hpp"

#include <boost/numeric/conversion/cast.hpp>
#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
// ProtoBuf
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>

#include <cmath>
#include <sstream>
#include <queue>
#include <fcntl.h>

#include <iostream>



#include "caffe2.pb.h"






namespace armnnCaffe2Parser{

using namespace armnn;
using namespace caffe2;
using namespace std;
using namespace google::protobuf::io;


namespace
{
    const float* GetArrayPtrFromBlob(const caffe2::Argument arg)
    {
        BOOST_ASSERT(arg.name()=="values");
        
        const float* arrayPtr = arg.floats().data();
        return arrayPtr;
    }

    void GetDataFromBlob(const caffe2::Argument arg, std::vector<float>& outData)
    {
        BOOST_ASSERT(arg.name()=="values");

        size_t blobSize = boost::numeric_cast<size_t>(arg.floats_size());
        if (blobSize != outData.size())
        {
            throw ParseException(
                boost::str(
                    boost::format(
                        "Data blob  in layer %2% has an unexpected size. "
                        "Expected %3% elements but got %4% elements. %5%") %
                        arg.name() %
                        outData.size() %
                        blobSize %
                        CHECK_LOCATION().AsString()));
        }

        int outSizeInt = boost::numeric_cast<int>(outData.size());
        for(int i = 0 ; i < outSizeInt; ++i)
        {
            outData[static_cast<size_t>(i)] = arg.floats(i);
        }

    }



}


// const std::map<std::string, Caffe2ParserBase::OperationParsingFunction>
//     Caffe2ParserBase::ms_Caffe2OperatorToParsingFunctions = {
//     { "ReLU",       &Caffe2ParserBase::ParseReluLayer },
//     };
    
    Caffe2ParserBase::Caffe2ParserBase()
        :m_Network(nullptr,nullptr)
    {

    }

    Caffe2Parser::Caffe2Parser() : Caffe2ParserBase()
    {}

ICaffe2Parser* ICaffe2Parser::Create()
{
    return new Caffe2Parser();
}



BindingPointInfo Caffe2ParserBase::GetNetworkInputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "data", m_NetworkInputsBindingInfo);
}

BindingPointInfo Caffe2ParserBase::GetNetworkOutputBindingInfo(const std::string& name) const
{
    return GetBindingInfo(name, "output", m_NetworkOutputsBindingInfo);
}



std::pair<armnn::LayerBindingId, armnn::TensorInfo> Caffe2ParserBase::GetBindingInfo(const std::string& layerName,
    const char* bindingPointDesc,
    const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    auto it = nameToBindingInfo.find(layerName);
     if (it == nameToBindingInfo.end())
    {
        throw InvalidArgumentException(
            boost::str(
                boost::format(
                    "Unknown binding %1% for layer '%2%'. %3%") %
                    bindingPointDesc %
                    layerName %
                    CHECK_LOCATION().AsString()));
    }
    return it->second;
}




TensorInfo Caffe2ParserBase::ArgumentToTensorInfo(const caffe2::Argument& arg)
{
    BOOST_ASSERT(arg.name()=="shape");
    std::vector<unsigned int> shape;
    for(int j=0; j<arg.ints_size();++j)
    {
        shape.push_back(static_cast<unsigned int>(arg.ints(j)));
    }
    return TensorInfo(boost::numeric_cast<unsigned int>(shape.size()),shape.data(),DataType::Float32);
}

void Caffe2ParserBase::TrackBindingPoint(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo,
    const char* bindingPointDesc,
    std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
{
    const std::string layerName = layer->GetName();
    auto it = nameToBindingInfo.find(layerName);
    if (it == nameToBindingInfo.end())
    {
        nameToBindingInfo[layerName] = std::make_pair(id, tensorInfo);
    }
    else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Id %1% used by more than one %2% layer %3%") %
                    id %
                    bindingPointDesc %
                    CHECK_LOCATION().AsString()));
    }
}




void Caffe2ParserBase::TrackInputBinding(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, layer->GetName(), m_NetworkInputsBindingInfo);
}


void Caffe2ParserBase::SetArmnnOutputSlotForCaffe2Output(const std::string& caffe2OutputName, armnn::IOutputSlot& armnnOutputSlot)
{
    auto it = m_ArmnnOutputSlotForCaffe2Output.find(caffe2OutputName);
    if (it == m_ArmnnOutputSlotForCaffe2Output.end())
    {
        m_ArmnnOutputSlotForCaffe2Output[caffe2OutputName] = &armnnOutputSlot;
    }
    else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Attempting to add duplicate entry for Caffe top '%1%' %2%") %
                    caffe2OutputName %
                    CHECK_LOCATION().AsString()));
    }
}


armnn::IOutputSlot& Caffe2ParserBase::GetArmnnOutputSlotForCaffe2Output(const std::string& caffe2OutputName) const
{
    auto it = m_ArmnnOutputSlotForCaffe2Output.find(caffe2OutputName);
    if (it != m_ArmnnOutputSlotForCaffe2Output.end())
    {
        return *it->second;
    }
    else
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Could not find armnn output slot for Caffe2 output '%1%' %2%") %
                    caffe2OutputName %
                    CHECK_LOCATION().AsString()));
    }
}

void Caffe2ParserBase::ParseInputLayer()
{
    const armnn::LayerBindingId inputId=boost::numeric_cast<armnn::LayerBindingId>(
        m_NetworkInputsBindingInfo.size());
    armnn::IConnectableLayer* const inputLayer = m_Network->AddInputLayer(inputId,"data");
    armnn::TensorInfo inputTensorInfo;
    auto overrideIt = m_InputShapes.find("data");
    const armnn::TensorShape& overrideShape=overrideIt->second;
    inputTensorInfo.SetShape(overrideShape);
    TrackInputBinding(inputLayer, inputId, inputTensorInfo);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    SetArmnnOutputSlotForCaffe2Output("data", inputLayer->GetOutputSlot(0));
}

 void Caffe2ParserBase::ParseReluLayer(const caffe2::OperatorDef& op)
 {

     BOOST_ASSERT(op.type()=="Relu");
     ActivationDescriptor activationDescriptor;
     const string& name = op.type();
     activationDescriptor.m_Function = ActivationFunction::ReLu;
     const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
     IConnectableLayer* const activationLayer = m_Network->AddActivationLayer(activationDescriptor, name.c_str());
     GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(activationLayer->GetInputSlot(0));
      activationLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
      SetArmnnOutputSlotForCaffe2Output(op.output(0), activationLayer->GetOutputSlot(0));
 }


 void Caffe2ParserBase::ParseFCLayer(const caffe2::OperatorDef& op)
 {
     FullyConnectedDescriptor tensorFullyConnectedDescriptor;
     tensorFullyConnectedDescriptor.m_TransposeWeightMatrix=false;

     //the weights name is stored at index 1
     
     auto it = blobs.find(op.input(1));
     if(it == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in FC Layer")%
                    op.input(1).c_str()
                    ));
     }
     const caffe2::OperatorDef& w=*it->second;

     //the biases are stored at the index 2
    auto it1 = blobs.find(op.input(2));
     if(it1 == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in FC Layer")%
                    op.input(2).c_str()
                    ));
     }
     const caffe2::OperatorDef& b=*it1->second;

     const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
    //at the index 1 the data is stored
     const float* weightDataPtr = GetArrayPtrFromBlob(w.arg(1));
    //at the index 0 the shape info is defined
     ConstTensor weights(ArgumentToTensorInfo(w.arg(0)), weightDataPtr);

     tensorFullyConnectedDescriptor.m_BiasEnabled = true;

     const float* biasDataPtr = GetArrayPtrFromBlob(b.arg(1));
     ConstTensor biases(ArgumentToTensorInfo(b.arg(0)), biasDataPtr);
     armnn::IConnectableLayer* fullyConnectedLayer = m_Network->AddFullyConnectedLayer(tensorFullyConnectedDescriptor, weights, biases,op.type().c_str());
     //the output shape = M x shape of bias
     TensorInfo outputInfo({inputInfo.GetShape()[0],biases.GetNumDimensions()}, DataType::Float32);

     GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(fullyConnectedLayer->GetInputSlot(0));
     fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
     SetArmnnOutputSlotForCaffe2Output(op.output(0),fullyConnectedLayer->GetOutputSlot(0));

 }



void Caffe2ParserBase::AddConvLayerWithDepthwiseConv(const caffe2::OperatorDef& op,
                                            const armnn::Convolution2dDescriptor convDesc,
                                            unsigned int kernel)
{
    BOOST_ASSERT(op.type()=="conv");
    DepthwiseConvolution2dDescriptor desc;
    desc.m_PadLeft      = convDesc.m_PadLeft;
    desc.m_PadRight     = convDesc.m_PadRight;
    desc.m_PadTop       = convDesc.m_PadTop;
    desc.m_PadBottom    = convDesc.m_PadBottom;
    desc.m_StrideX      = convDesc.m_StrideX;
    desc.m_StrideY      = convDesc.m_StrideY;
    desc.m_BiasEnabled  = convDesc.m_BiasEnabled;

    auto it = blobs.find(op.input(1));
     if(it == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in conv Layer")%
                    op.input(2).c_str()
                    ));
     }
     const caffe2::OperatorDef& w = *it->second;

     unsigned int numFilters = boost::numeric_cast<unsigned int>(w.arg(0).ints(0));

    const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();

    caffe2::Argument outputShape;
    outputShape.set_name("shape");
    outputShape.add_ints(0);
    outputShape.set_ints(0, inputInfo.GetShape()[0]);
    outputShape.add_ints(1);
    outputShape.set_ints(1, numFilters);
    outputShape.add_ints(2);
    outputShape.set_ints(
        2, (static_cast<int>(
                static_cast<float>(inputInfo.GetShape()[2] + 2 * desc.m_PadBottom - kernel) /
                static_cast<float>(desc.m_StrideY)) + 1));
    outputShape.add_ints(3);
    outputShape.set_ints(
        3, (static_cast<int>(
                static_cast<float>(inputInfo.GetShape()[3] + 2 * desc.m_PadRight - kernel) /
                static_cast<float>(desc.m_StrideX)) + 1));

     size_t allWeightsSize = boost::numeric_cast<size_t>(w.arg(0).ints(0) * kernel * kernel);
    vector<float> weightData(allWeightsSize);
    GetDataFromBlob(w.arg(1), weightData);
    armnn::IConnectableLayer* returnLayer = nullptr;
    ConstTensor weights(ArgumentToTensorInfo(w.arg(0)),weightData.data());

    if(desc.m_BiasEnabled)
    {

        TensorInfo biasInfo ;
        auto it = blobs.find(op.input(2));
        if(it == blobs.end())
        {
           
            throw ParseException(
                boost::str(
                    boost::format(
                        "Could not find the '%1%' in conv Layer")%
                        op.input(2).c_str()
                        ));
        }
        const caffe2::OperatorDef& b = *it->second;

        vector<float> biasData;
        biasData.resize(boost::numeric_cast<size_t>(outputShape.ints(1)), 1.f);
        GetDataFromBlob(b.arg(1),biasData);
        biasInfo = ArgumentToTensorInfo(b.arg(0));
        ConstTensor biases(biasInfo, biasData.data());

        returnLayer = m_Network->AddDepthwiseConvolution2dLayer(desc, weights, biases, op.type().c_str());

    }

    else
    {
        returnLayer = m_Network->AddDepthwiseConvolution2dLayer(desc, weights, op.type().c_str());
    }


    if (!returnLayer)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to create depthwise convolution layer. "
                    "Layer=%1% #filters=%2% %3%") %
                    op.type() %
                    numFilters %
                    CHECK_LOCATION().AsString()));
    }

    armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffe2Output(op.input(0));
    inputConnection.Connect(returnLayer->GetInputSlot(0));
    returnLayer->GetOutputSlot(0).SetTensorInfo(ArgumentToTensorInfo(outputShape));
    SetArmnnOutputSlotForCaffe2Output(op.output(0),returnLayer->GetOutputSlot(0));

}



 void Caffe2ParserBase::ParseConvLayer(const caffe2::OperatorDef& op)
 {

     
     BOOST_ASSERT(op.type()=="Conv");
     //create a map of arg name and arg
     
     std::map<std::string, const caffe2::Argument*> args;
     for(int i=0; i<op.arg_size(); ++i)
     {
         args.insert({op.arg(i).name(),&op.arg(i)});
     }
     auto it = blobs.find(op.input(1));
     if(it == blobs.end())
     {
         throw ParseException(
            boost::str(
                boost::format(
                    "Could not find the '%1%' in conv Layer")%
                    op.input(2).c_str()
                    ));
     }
     const caffe2::OperatorDef& w = *it->second;

     unsigned int numFilters = boost::numeric_cast<unsigned int>(w.arg(0).ints(0));

      auto it1 = args.find("group");
     
     unsigned int numGroups = 1;
     if(it1!=args.end())
     {
         const caffe2::Argument& a = *it1->second;
         numGroups = boost::numeric_cast<unsigned int>(a.i());
     }
      std::cout<<"group "<<numGroups<<std::endl;

     unsigned int kernel = 0;
     auto it2 = args.find("kernel");
     if(it2!=args.end())
     {
         const caffe2::Argument& a = *it2->second;
         kernel = boost::numeric_cast<unsigned int>(a.i());
     }
     std::cout<<"kernel "<<kernel<<std::endl;

     unsigned int stride = 1;
     auto it3 = args.find("stride");
     if(it3!=args.end())
     {
         const caffe2::Argument& a = *it3->second;
         stride = boost::numeric_cast<unsigned int>(a.i());
     }

     std::cout<<"stride "<<stride<<std::endl;

     unsigned int pad = 0;
     auto it4 = args.find("pad");
     if(it4!=args.end())
     {
         const caffe2::Argument& a = *it4->second;
         pad = boost::numeric_cast<unsigned int>(a.i());
     }

     std::cout<<"pad "<<pad<<std::endl;
     Convolution2dDescriptor convolution2dDescriptor;

     convolution2dDescriptor.m_PadLeft = pad;
     convolution2dDescriptor.m_PadRight = pad;
     convolution2dDescriptor.m_PadTop = pad;
     convolution2dDescriptor.m_PadBottom = pad;
     convolution2dDescriptor.m_StrideX = stride;
     convolution2dDescriptor.m_StrideY = stride;
     convolution2dDescriptor.m_BiasEnabled = op.input_size()==3 ? true : false;
    

    if (numGroups > numFilters)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Error parsing Convolution: %1%. "
                    "The 'group'=%2% parameter cannot be larger than the "
                    "number of filters supplied ='%3%'. %4%") %
                    op.name() %
                    numGroups %
                    numFilters %
                    CHECK_LOCATION().AsString()));
    }

    
      armnn::IOutputSlot& inputConnection = GetArmnnOutputSlotForCaffe2Output(op.input(0));
      
    const TensorInfo& inputInfo = inputConnection.GetTensorInfo();
   
     if (inputInfo.GetNumDimensions() != 4)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Convolution input shape is expected to have 4 dimensions. "
                    "%1%'s input has only %2%. %3%") %
                    op.name() %
                    inputInfo.GetNumDimensions() %
                    CHECK_LOCATION().AsString()));
    }


    if (numGroups > 1)
    {
        if (numGroups > inputInfo.GetShape()[1])
        {
            throw ParseException(
                boost::str(
                    boost::format(
                        "Error parsing Convolution: %1%. "
                        "The 'group'=%2% parameter cannot be larger than the "
                        "channel of the input shape=%3% (in NCHW format). %4%") %
                        op.name() %
                        numGroups %
                        inputInfo.GetShape()[1] %
                        CHECK_LOCATION().AsString()));
        }

    }
    else if (numGroups == inputInfo.GetShape()[1])
        {
            // we use a depthwise convolution here, because the number of groups equals to the
            // input channels
            AddConvLayerWithDepthwiseConv(op, convolution2dDescriptor, kernel);
            return;
        }
   
    caffe2::Argument outputShape;
    outputShape.set_name("shape");
    outputShape.add_ints(0);
    outputShape.set_ints(0,inputInfo.GetShape()[0]);
    outputShape.add_ints(1);
    outputShape.set_ints(1,numFilters);
    outputShape.add_ints(2);
    outputShape.set_ints(
        2, (static_cast<int>(static_cast<float>(inputInfo.GetShape()[2]) +  2 * pad - kernel)/
                            static_cast<float>(stride))+1);
    outputShape.add_ints(3);
    outputShape.set_ints(
        3, (static_cast<int>(static_cast<float>(inputInfo.GetShape()[2]) +  2 * pad - kernel)/
                            static_cast<float>(stride))+1);
    
    vector<float> weightData(boost::numeric_cast<size_t>(w.arg(0).ints(0) *
                                                        w.arg(0).ints(1) *
                                                        w.arg(0).ints(2) *
                                                        w.arg(0).ints(3)));
    
    GetDataFromBlob(w.arg(1),weightData);

    armnn::IConnectableLayer* returnLayer = nullptr;

    ConstTensor weights(ArgumentToTensorInfo(w.arg(0)),weightData.data());

    if (convolution2dDescriptor.m_BiasEnabled)
    {
         TensorInfo biasInfo ;
        auto it = blobs.find(op.input(2));
        if(it == blobs.end())
        {
           
            throw ParseException(
                boost::str(
                    boost::format(
                        "Could not find the '%1%' in conv Layer")%
                        op.input(2).c_str()
                        ));
        }
        const caffe2::OperatorDef& b = *it->second;

        vector<float> biasData;
        biasData.resize(boost::numeric_cast<size_t>(outputShape.ints(1)), 1.f);
        GetDataFromBlob(b.arg(1),biasData);
        biasInfo = ArgumentToTensorInfo(b.arg(0));
        ConstTensor biases(biasInfo, biasData.data());

        returnLayer = m_Network->AddConvolution2dLayer(convolution2dDescriptor, weights, biases, op.type().c_str());

    }
    else
    {
        returnLayer = m_Network->AddConvolution2dLayer(convolution2dDescriptor, weights, op.type().c_str());
    }
   
  
    inputConnection.Connect(returnLayer->GetInputSlot(0));
    returnLayer->GetOutputSlot(0).SetTensorInfo(ArgumentToTensorInfo(outputShape));

    if (!returnLayer)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to create Convolution layer. "
                    "Layer=%1% #groups=%2% #filters=%3% %4%") %
                    op.name() %
                    numGroups %
                    numFilters %
                    CHECK_LOCATION().AsString()));
    }
    std::cout<<"output "<<op.output(0)<<std::endl;
    SetArmnnOutputSlotForCaffe2Output(op.output(0), returnLayer->GetOutputSlot(0));

 }



void Caffe2ParserBase::TrackOutputBinding(armnn::IConnectableLayer* layer,
    armnn::LayerBindingId id,
    const armnn::TensorInfo& tensorInfo)
{
    return TrackBindingPoint(layer, id, tensorInfo, layer->GetName(), m_NetworkOutputsBindingInfo);
}


void Caffe2ParserBase::LoadNetDef(caffe2::NetDef& init,caffe2::NetDef& predict)
{
    //Create a lookup of Caff2 layers by output name
    for (int i=0;i<predict.op_size(); ++i)
    {
        const caffe2::OperatorDef& op=predict.op(i);
        
        for(int i=0 ; i<op.output_size();++i)
        {
            m_Caffe2OperatorsByOutputName[op.output(i)]=&op;

        }

    }

    std::vector<const caffe2::OperatorDef*> nodes;
    for(int i=0;i<predict.op_size();i++)
    {
        nodes.push_back(&predict.op(i));
    }

    //stores the corresponding name and index of blobs in init_net
    for(int i=0;i<init.op_size();++i)
    {
        blobs.insert({init.op(i).output(0),&init.op(i)});
    }

    this->ParseInputLayer();
    
    OperationParsingFunction fun = &Caffe2ParserBase::ParseConvLayer;
    (this->*fun)(*nodes.at(0));

    OperationParsingFunction funr = &Caffe2ParserBase::ParseReluLayer;
    (this->*funr)(*nodes.at(1));
  
    for (const std::string& requestedOutput : m_RequestedOutputs)
    {
        armnn::IOutputSlot& outputSlot = GetArmnnOutputSlotForCaffe2Output(requestedOutput);

        const armnn::LayerBindingId outputId = boost::numeric_cast<armnn::LayerBindingId>(
            m_NetworkOutputsBindingInfo.size());
        armnn::IConnectableLayer* const outputLayer = m_Network->AddOutputLayer(outputId, requestedOutput.c_str());
        outputSlot.Connect(outputLayer->GetInputSlot(0));

        TrackOutputBinding(outputLayer, outputId, outputLayer->GetInputSlot(0).GetConnection()->GetTensorInfo());
    }



}

armnn::INetworkPtr Caffe2Parser::CreateNetworkFromBinaryFile(const char* predict_net,const char* init_net,const std::map<std::string, armnn::TensorShape>& inputShapes,
                                                    const std::vector<std::string>& requestedOutputs)
{
    //reading the predict net
    FILE* fd = fopen(predict_net, "rb");
    
    if (fd == nullptr)
    {
        throw FileNotFoundException(
            boost::str(
                boost::format(
                    "Failed to open predict_net file at: %1% %2%") %
                    predict_net %
                    CHECK_LOCATION().AsString()));
    }
     
    // Parses the file into a message.
    NetDef predict;

    FileInputStream  inStream(fileno(fd));
    CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX, INT_MAX);
    bool success = predict.ParseFromCodedStream(&codedStream);
    fclose(fd);

    if (!success)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to parse predict net protobuf file: %1% %2%") %
                    predict_net %
                    CHECK_LOCATION().AsString()));
    }

    //reading the init net



    FILE* fd1 = fopen(init_net, "rb");

    if (fd1 == nullptr)
    {
        throw FileNotFoundException(
            boost::str(
                boost::format(
                    "Failed to open init_net file at: %1% %2%") %
                    init_net %
                    CHECK_LOCATION().AsString()));
    }

    // Parses the file into a message.
    NetDef init;

    FileInputStream  inStream1(fileno(fd1));
    CodedInputStream codedStream1(&inStream1);
    codedStream1.SetTotalBytesLimit(INT_MAX, INT_MAX);
    bool success1 = init.ParseFromCodedStream(&codedStream1);
    fclose(fd1);

    if (!success1)
    {
        throw ParseException(
            boost::str(
                boost::format(
                    "Failed to parse init net protobuf file: %1% %2%") %
                    init_net %
                    CHECK_LOCATION().AsString()));
    }


    return CreateNetworkFromNetDef(init,predict,inputShapes,requestedOutputs);
}
armnn::INetworkPtr Caffe2ParserBase::CreateNetworkFromNetDef(caffe2::NetDef& init,caffe2::NetDef& predict,const std::map<std::string, armnn::TensorShape>& inputShapes,
                                                const std::vector<std::string>& requestedOutputs)
{


    m_NetworkInputsBindingInfo.clear();

    m_Network=INetwork::Create();
 
  
    m_InputShapes=inputShapes;

    if (requestedOutputs.size() == 0)
    {
        throw ParseException("requestedOutputs must have at least one entry");
    }
    m_RequestedOutputs = requestedOutputs;

    try
    {
        LoadNetDef(init,predict);

    }catch(const ParseException& e)
    {
        throw e;
    }

    return move(m_Network);

}

}

