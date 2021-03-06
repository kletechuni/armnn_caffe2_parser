﻿//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../Cifar10Database.hpp"
#include "armnnTfParser/ITfParser.hpp"

int main(int argc, char* argv[])
{
    armnn::TensorShape inputTensorShape({ 1, 32, 32, 3 });

    int retVal = EXIT_FAILURE;
    try
    {
        using DataType = float;
        using DatabaseType = Cifar10Database;
        using ParserType = armnnTfParser::ITfParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                     argc, argv, "cifar10_tf.prototxt", false,
                     "data", "prob", { 0, 1, 2, 4, 7 },
                     [](const char* dataDir, const ModelType&) {
                         return DatabaseType(dataDir, true);
                     }, &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: TfCifar10-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
