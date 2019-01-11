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

<<<<<<< HEAD
=======
#include <iostream>

>>>>>>> vinay


#include "caffe2.pb.h"


namespace armnnCaffe2Parser{

using namespace armnn;
using namespace caffe2;
using namespace std;
using namespace google::protobuf::io;


<<<<<<< HEAD
const std::map<std::string, Caffe2ParserBase::OperationParsingFunction>
    Caffe2ParserBase::ms_Caffe2OperatorToParsingFunctions ={{}};

ICaffe2Parser* ICaffe2Parser::Create()
{
    return new Caffe2ParserBase();
}

void Caffe2ParserBase::LoadNetDef(caffe2::NetDef& netDef)
{
    //Create a lookup of Caff2 layers by output name
    for (int i=0;i<netDef.op_size(); ++i)
    {
        const caffe2::OperatorDef& op=netDef.op(i);
        for(int i=0 ; i<op.output_size;++i)
=======
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
                    "Could not find armnn output slot for Caffe top '%1%' %2%") %
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
     ActivationDescriptor activationDescriptor;
     const string& name = op.type();
     const TensorInfo& inputInfo = GetArmnnOutputSlotForCaffe2Output(op.input(0)).GetTensorInfo();
     IConnectableLayer* const activationLayer = m_Network->AddActivationLayer(activationDescriptor, name.c_str());
     GetArmnnOutputSlotForCaffe2Output(op.input(0)).Connect(activationLayer->GetInputSlot(0));
      activationLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
      SetArmnnOutputSlotForCaffe2Output(op.output(0), activationLayer->GetOutputSlot(0));
 }





void Caffe2ParserBase::LoadNetDef(caffe2::NetDef& init,caffe2::NetDef& predict)
{
    //Create a lookup of Caff2 layers by output name
    for (int i=0;i<predict.op_size(); ++i)
    {
        const caffe2::OperatorDef& op=predict.op(i);
        
        for(int i=0 ; i<op.output_size();++i)
>>>>>>> vinay
        {
            m_Caffe2OperatorsByOutputName[op.output(i)]=&op;

        }

    }

    std::vector<const caffe2::OperatorDef*> nodes;
<<<<<<< HEAD
    for(int i=0;i<netDef.op_size();i++)
    {
        nodes.push_back(&netDef.op(i));
    }
}

void Caffe2ParserBase::CreateNetworkFromBinaryFile(const char* graphFile)
{
    FILE* fd = fopen(graphFile, "rb");

=======
    for(int i=0;i<predict.op_size();i++)
    {
        nodes.push_back(&predict.op(i));
    }
    ParseInputLayer();
}

void Caffe2Parser::CreateNetworkFromBinaryFile(const char* predict_net,const char* init_net,const std::map<std::string, armnn::TensorShape>& inputShapes)
{
    //reading the predict net
    FILE* fd = fopen(predict_net, "rb");
    
>>>>>>> vinay
    if (fd == nullptr)
    {
        throw FileNotFoundException(
            boost::str(
                boost::format(
<<<<<<< HEAD
                    "Failed to open graph file at: %1% %2%") %
                    graphFile %
                    CHECK_LOCATION().AsString()));
    }

    // Parses the file into a message.
    NetDef netParam;
=======
                    "Failed to open predict_net file at: %1% %2%") %
                    predict_net %
                    CHECK_LOCATION().AsString()));
    }
     
    // Parses the file into a message.
    NetDef predict;
>>>>>>> vinay

    FileInputStream  inStream(fileno(fd));
    CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX, INT_MAX);
<<<<<<< HEAD
    bool success = netParam.ParseFromCodedStream(&codedStream);
=======
    bool success = predict.ParseFromCodedStream(&codedStream);
>>>>>>> vinay
    fclose(fd);

    if (!success)
    {
        throw ParseException(
            boost::str(
                boost::format(
<<<<<<< HEAD
                    "Failed to parse protobuf file: %1% %2%") %
                    graphFile %
                    CHECK_LOCATION().AsString()));
    }

}


void Caffe2ParserBase::CreateNetworkFromNetDef(caffe2::NetDef& netDef)
{
    m_Network=INetwork::Create();
    try
    {
=======
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


    CreateNetworkFromNetDef(init,predict,inputShapes);

}


void Caffe2ParserBase::CreateNetworkFromNetDef(caffe2::NetDef& init,caffe2::NetDef& predict,const std::map<std::string, armnn::TensorShape>& inputShapes)
{
     m_NetworkInputsBindingInfo.clear();


    m_Network=INetwork::Create();
    m_InputShapes=inputShapes;
    try
    {
        LoadNetDef(init,predict);
>>>>>>> vinay

    }catch(const ParseException& e)
    {
        throw e;
    }

}

<<<<<<< HEAD


=======
>>>>>>> vinay
}

