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



#include "caffe2.pb.h"


namespace armnnCaffe2Parser{

using namespace armnn;
using namespace caffe2;
using namespace std;
using namespace google::protobuf::io;


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
        {
            m_Caffe2OperatorsByOutputName[op.output(i)]=&op;

        }

    }

    std::vector<const caffe2::OperatorDef*> nodes;
    for(int i=0;i<netDef.op_size();i++)
    {
        nodes.push_back(&netDef.op(i));
    }
}

void Caffe2ParserBase::CreateNetworkFromBinaryFile(const char* graphFile)
{
    FILE* fd = fopen(graphFile, "rb");

    if (fd == nullptr)
    {
        throw FileNotFoundException(
            boost::str(
                boost::format(
                    "Failed to open graph file at: %1% %2%") %
                    graphFile %
                    CHECK_LOCATION().AsString()));
    }

    // Parses the file into a message.
    NetDef netParam;

    FileInputStream  inStream(fileno(fd));
    CodedInputStream codedStream(&inStream);
    codedStream.SetTotalBytesLimit(INT_MAX, INT_MAX);
    bool success = netParam.ParseFromCodedStream(&codedStream);
    fclose(fd);

    if (!success)
    {
        throw ParseException(
            boost::str(
                boost::format(
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

    }catch(const ParseException& e)
    {
        throw e;
    }

}



}

