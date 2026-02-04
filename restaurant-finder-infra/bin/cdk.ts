#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { BaseStackProps } from "../lib/types";
import { DockerImageStack, AgentCoreStack } from "../lib/stacks";

const app = new cdk.App();

// Check for existing image URI to skip Docker build
// Usage: npx cdk deploy restaurantFinder-AgentCoreStack -c imageUri=<existing-uri>
const existingImageUri = app.node.tryGetContext("imageUri") as string | undefined;

const deploymentProps: BaseStackProps = {
  appName: "restaurantFinder",
  /* If you don't specify 'env', this stack will be environment-agnostic.
   * Account/Region-dependent features and context lookups will not work,
   * but a single synthesized template can be deployed anywhere. */

  /* Uncomment the next line to specialize this stack for the AWS Account
   * and Region that are implied by the current CLI configuration. */
  // env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },

  /* Uncomment the next line if you know exactly what Account and Region you
   * want to deploy the stack to. */
  // env: { account: '123456789012', region: 'us-east-1' },

  /* For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html */
};

if (existingImageUri) {
  // Skip Docker build - use existing image URI
  // This is useful for iterating on infrastructure without rebuilding the image
  console.log(`Using existing image URI: ${existingImageUri}`);

  new AgentCoreStack(
    app,
    `restaurantFinder-AgentCoreStack`,
    {
      ...deploymentProps,
      imageUri: existingImageUri,
    },
  );
} else {
  // Build Docker image and deploy both stacks
  const dockerImageStack = new DockerImageStack(
    app,
    `restaurantFinder-DockerImageStack`,
    deploymentProps,
  );

  const agentCoreStack = new AgentCoreStack(
    app,
    `restaurantFinder-AgentCoreStack`,
    {
      ...deploymentProps,
      imageUri: dockerImageStack.imageUri,
    },
  );

  agentCoreStack.addDependency(dockerImageStack);
}
